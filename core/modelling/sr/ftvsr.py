import torch
import torch.nn as nn
import torch.nn.functional as F
from core.modelling.sr.ftt import FTT
from core.modelling.dct.dct import check_and_padding_imgs
from core.modelling.optflow.spynet import SPyNet, flow_warp
from core.modelling.utils.pixelshuffle import PixelShufflePack
from core.modelling.utils.residual import ResidualBlocksWithInputConv


class FTVSRNet(nn.Module):
    def __init__(self,
                 mid_channels=64,
                 num_blocks=60,
                 stride=4,
                 keyframe_stride=3,
                 dct_kernel=(8,8),
                 d_model=512,
                 n_heads=8):
        super().__init__()

        self.dct_kernel = dct_kernel
        self.mid_channels = mid_channels
        self.keyframe_stride = keyframe_stride
        self.stride = stride
        self.spynet = SPyNet()
        self.feat_extractor = ResidualBlocksWithInputConv(3, mid_channels, 5)  
        self.feat_extractor_resids = ResidualBlocksWithInputConv(3, mid_channels, 5)
        self.fusion_resids = nn.Conv2d(2 * mid_channels, mid_channels, 1, 1, 0, bias=True)
        self.LTAM = LTAM(stride = self.stride)
        self.resblocks = ResidualBlocksWithInputConv(2 * mid_channels, mid_channels, num_blocks)
        self.fusion = nn.Conv2d(3 * mid_channels, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.FTT = FTT(dct_kernel=dct_kernel, d_model=d_model, n_heads=n_heads)


    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].view(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].view(-1, c, h, w)
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, to_cpu=False):
        n, t, c, h, w = lrs.size()

        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        outputs = self.feat_extractor(lrs.view(-1,c,h,w)).view(n,t,-1,h,w)
        outputs = torch.unbind(outputs, dim=1)
        outputs = list(outputs)
        keyframe_idx_forward = list(range(0, t, self.keyframe_stride))
        keyframe_idx_backward = list(range(t-1, 0, 0-self.keyframe_stride))
        # backward-time propgation
        feat_buffers = []
        sparse_feat_buffers_s1 = []
        sparse_feat_buffers_s2 = []
        sparse_feat_buffers_s3 = []
        index_feat_buffers_s1 = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        # grid_y, grid_x = torch.meshgrid(torch.arange(0, h//self.stride), torch.arange(0, w//self.stride))
        if 'indexing' in torch.meshgrid.__code__.co_varnames:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, h//self.stride),
                torch.arange(0, w//self.stride),
                indexing='ij')
        else:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, h//self.stride),
                torch.arange(0, w//self.stride))


        location_update = torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)

        first_iter = True
        for i in range(t - 1, -1, -1):
            lr_curr_feat = outputs[i]
            if not first_iter: # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1),padding_mode='border')
                
                # refresh the location map
                # flow = F.adaptive_avg_pool2d(flow, (h//self.stride, w//self.stride)) / self.stride
                flow = F.avg_pool2d(flow, kernel_size=self.stride, stride=self.stride) / self.stride
                # flow = F.interpolate(flow, (h//self.stride, w//self.stride), mode='nearest') / self.stride # TODO: new

                location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),padding_mode='border',interpolation="nearest")# n , 2t , h , w

                # set the real feature
                sparse_feat_buffer_s1 = torch.stack(sparse_feat_buffers_s1, dim=1)
                sparse_feat_buffer_s2 = torch.stack(sparse_feat_buffers_s2, dim=1)
                sparse_feat_buffer_s3 = torch.stack(sparse_feat_buffers_s3, dim=1)
                index_feat_buffer_s1 = torch.stack(index_feat_buffers_s1, dim=1)
                feat_prop = self.LTAM(lr_curr_feat,index_feat_buffer_s1,feat_prop,sparse_feat_buffer_s1,sparse_feat_buffer_s2,sparse_feat_buffer_s3,location_update)

                # add the location map
                if i in keyframe_idx_backward:
                    location_update = torch.cat([location_update,torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)],dim=1) # n , 2t , h , w
            first_iter = False

            feat_prop = torch.cat([lr_curr_feat,feat_prop], dim=1)
            feat_prop = self.resblocks(feat_prop)
            feat_buffers.append(feat_prop)
            if i in keyframe_idx_backward:
                # cross-scale feature * 4
                sparse_feat_prop_s1 = F.unfold(feat_prop, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                sparse_feat_prop_s1 = F.fold(sparse_feat_prop_s1, output_size=(int(h//self.stride), int(w//self.stride)), kernel_size=(1,1), padding=0, stride=1)
                sparse_feat_buffers_s1.append(sparse_feat_prop_s1)

                index_feat_prop_s1 = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                index_feat_prop_s1 = F.fold(index_feat_prop_s1, output_size=(int(h//self.stride), int(w//self.stride)), kernel_size=(1,1), padding=0, stride=1) 
                index_feat_buffers_s1.append(index_feat_prop_s1)

                # cross-scale feature * 8
                sparse_feat_prop_s2 = F.unfold(feat_prop, kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=int(0.25*self.stride), stride=self.stride) 
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(int(1.5*h),int(1.5*w)), kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=0, stride=int(1.5*self.stride))
                # sparse_feat_prop_s2 = F.adaptive_avg_pool2d(sparse_feat_prop_s2, (int(h), int(w)))
                sparse_feat_prop_s2 = F.interpolate(sparse_feat_prop_s2, (int(h), int(w)), mode='bilinear', align_corners=False) # TODO: new
                sparse_feat_prop_s2 = F.unfold(sparse_feat_prop_s2, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(int(h//self.stride), int(w//self.stride)), kernel_size=(1,1), padding=0, stride=1)
                sparse_feat_buffers_s2.append(sparse_feat_prop_s2)

                # cross-scale feature * 12
                sparse_feat_prop_s3 = F.unfold(feat_prop, kernel_size=(int(2*self.stride),int(2*self.stride)), padding=int(0.5*self.stride), stride=self.stride) 
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(int(2*h),int(2*w)), kernel_size=(int(2*self.stride),int(2*self.stride)), padding=0, stride=int(2*self.stride))
                # sparse_feat_prop_s3 = F.adaptive_avg_pool2d(sparse_feat_prop_s3, (int(h), int(w)))
                sparse_feat_prop_s3 = F.interpolate(sparse_feat_prop_s3, (int(h), int(w)), mode='bilinear', align_corners=False) # TODO: new
                sparse_feat_prop_s3 = F.unfold(sparse_feat_prop_s3, kernel_size=(self.stride, self.stride), padding=0, stride=self.stride) 
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(int(h//self.stride), int(w//self.stride)), kernel_size=(1,1), padding=0, stride=1)
                sparse_feat_buffers_s3.append(sparse_feat_prop_s3)

        outputs_back = feat_buffers[::-1]
        del location_update
        del feat_buffers
        del sparse_feat_buffers_s1
        del sparse_feat_buffers_s2
        del sparse_feat_buffers_s3
        del index_feat_buffers_s1

        # forward-time propagation and upsampling
        fina_out = []
        bicubic_imgs = []
        feat_buffers = []
        sparse_feat_buffers_s1 = []
        sparse_feat_buffers_s2 = []
        sparse_feat_buffers_s3 = []
        index_feat_buffers_s1 = []

        feat_prop = torch.zeros_like(feat_prop)
        # grid_y, grid_x = torch.meshgrid(torch.arange(0, h//self.stride), torch.arange(0, w//self.stride))
        if 'indexing' in torch.meshgrid.__code__.co_varnames:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, h//self.stride),
                torch.arange(0, w//self.stride),
                indexing='ij')
        else:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, h//self.stride),
                torch.arange(0, w//self.stride))

        location_update = torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            lr_curr_feat = outputs[i]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1),padding_mode='border')

                # refresh the location map
                # flow = F.adaptive_avg_pool2d(flow,(h//self.stride, w//self.stride))/self.stride
                flow = F.avg_pool2d(flow, kernel_size=self.stride, stride=self.stride) / self.stride
                # flow = F.interpolate(flow, (h//self.stride, w//self.stride), mode='nearest') / self.stride # TODO: new

                location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),padding_mode='border',interpolation="nearest")# n , 2t , h , w

                # set the real feature
                sparse_feat_buffer_s1 = torch.stack(sparse_feat_buffers_s1, dim=1)
                sparse_feat_buffer_s2 = torch.stack(sparse_feat_buffers_s2, dim=1)
                sparse_feat_buffer_s3 = torch.stack(sparse_feat_buffers_s3, dim=1)

                index_feat_buffer_s1 = torch.stack(index_feat_buffers_s1, dim=1)
                feat_prop = self.LTAM(lr_curr_feat,index_feat_buffer_s1,feat_prop,sparse_feat_buffer_s1,sparse_feat_buffer_s2,sparse_feat_buffer_s3,location_update)

                # add the location map
                if i in keyframe_idx_forward:
                    location_update = torch.cat([location_update,torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)],dim=1)

            feat_prop = torch.cat([outputs[i], feat_prop], dim=1)
            feat_prop = self.resblocks(feat_prop)
            feat_buffers.append(feat_prop)

            if i in keyframe_idx_forward:
                # cross-scale feature * 4
                sparse_feat_prop_s1 = F.unfold(feat_prop, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                sparse_feat_prop_s1 = F.fold(sparse_feat_prop_s1, output_size=(int(h//self.stride), int(w//self.stride)), kernel_size=(1,1), padding=0, stride=1)
                sparse_feat_buffers_s1.append(sparse_feat_prop_s1)

                index_feat_prop_s1 = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                index_feat_prop_s1 = F.fold(index_feat_prop_s1, output_size=(int(h//self.stride), int(w//self.stride)), kernel_size=(1,1), padding=0, stride=1)
                index_feat_buffers_s1.append(index_feat_prop_s1)

                # cross-scale feature * 8
                sparse_feat_prop_s2 = F.unfold(feat_prop, kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=int(0.25*self.stride), stride=self.stride) 
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(int(1.5*h),int(1.5*w)), kernel_size=(int(1.5*self.stride),int(1.5*self.stride)), padding=0, stride=int(1.5*self.stride))
                # sparse_feat_prop_s2 = F.adaptive_avg_pool2d(sparse_feat_prop_s2, (int(h), int(w)))
                sparse_feat_prop_s2 = F.interpolate(sparse_feat_prop_s2, (int(h), int(w)), mode='bilinear', align_corners=False) # TODO: new
                sparse_feat_prop_s2 = F.unfold(sparse_feat_prop_s2, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(int(h//self.stride), int(w//self.stride)), kernel_size=(1,1), padding=0, stride=1) 
                sparse_feat_buffers_s2.append(sparse_feat_prop_s2)

                # cross-scale feature * 12
                sparse_feat_prop_s3 = F.unfold(feat_prop, kernel_size=(int(2*self.stride),int(2*self.stride)), padding=int(0.5*self.stride), stride=self.stride) 
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(int(2*h),int(2*w)), kernel_size=(int(2*self.stride),int(2*self.stride)), padding=0, stride=int(2*self.stride))
                # sparse_feat_prop_s3 = F.adaptive_avg_pool2d(sparse_feat_prop_s3, (int(h), int(w)))
                sparse_feat_prop_s3 = F.interpolate(sparse_feat_prop_s3, (int(h), int(w)), mode='bilinear', align_corners=False) # TODO: new
                sparse_feat_prop_s3 = F.unfold(sparse_feat_prop_s3, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
                sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(int(h//self.stride), int(w//self.stride)), kernel_size=(1,1), padding=0, stride=1)
                sparse_feat_buffers_s3.append(sparse_feat_prop_s3)

            # upsampling given the backward and forward features
            out = torch.cat([outputs_back[i],lr_curr_feat,feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))

            # # add residuals
            # resid_curr = 2.0 * resids[:, i, :, :, :] - 1.0
            # resid_curr_features = self.feat_extractor_resids(resid_curr)
            # out = self.lrelu(self.fusion_resids(torch.cat([out, resid_curr_features], dim=1)))

            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            bicubic_imgs.append(base)
            out += base

            fina_out.append(out)

        del location_update
        del sparse_feat_buffers_s1
        del sparse_feat_buffers_s2
        del sparse_feat_buffers_s3
        del index_feat_buffers_s1

        #frequency attention
        high_frequency_out = torch.stack(fina_out, dim=1) #n,t,c,h,w
        bicubic_imgs = torch.stack(bicubic_imgs, dim=1) # n,t,c,h,w

        #padding images
        bicubic_imgs, padding_h, padding_w = check_and_padding_imgs(bicubic_imgs, self.dct_kernel)
        high_frequency_imgs, _, _ = check_and_padding_imgs(high_frequency_out, self.dct_kernel)

        n,t,c,h,w = bicubic_imgs.shape
        flows_forward, flows_backward = self.compute_flow(high_frequency_imgs)

        final_out = self.FTT(
            bicubic_imgs, high_frequency_imgs, 
            [flows_forward, flows_backward], 
            [padding_h, padding_w], to_cpu)

        return final_out


class LTAM(nn.Module):
    def __init__(self, stride=4):
        super().__init__()
        self.stride = stride
        self.fusion = nn.Conv2d(3 * 64, 64, 3, 1, 1, bias=True)

    def forward(self, curr_feat, index_feat_set_s1 , anchor_feat, sparse_feat_set_s1 ,sparse_feat_set_s2, sparse_feat_set_s3, location_feat):
        """
        input :   anchor_feat  # n * c * h * w
        input :   sparse_feat_set_s1      # n * t * (c*4*4) * (h//4) * (w//4) 
        input :   sparse_feat_set_s2      # n * t * (c*4*4) * (h//4) * (w//4)
        input :   sparse_feat_set_s3      # n * t * (c*4*4) * (h//4) * (w//4)
        input :   location_feat  #  n * (2*t) * (h//4) * (w//4)
        output :   fusion_feature  # n * c * h * w
        """
        n, c, h, w = anchor_feat.size()
        t = sparse_feat_set_s1.size(1)

        feat_len = int(c*self.stride*self.stride)
        feat_num = int((h//self.stride) * (w//self.stride))

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1]
        grid_flow = location_feat.contiguous().view(n,t,2,h//self.stride,w//self.stride).permute(0, 1, 3, 4, 2)
        grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / max(w//self.stride, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / max(h//self.stride, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=4)

        output_s1 = F.grid_sample(sparse_feat_set_s1.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s2 = F.grid_sample(sparse_feat_set_s2.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s3 = F.grid_sample(sparse_feat_set_s3.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
     
        index_output_s1 = F.grid_sample(index_feat_set_s1.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)

        curr_feat = F.unfold(curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride) 
        curr_feat = curr_feat.permute(0, 2, 1)
        curr_feat = F.normalize(curr_feat, dim=2).unsqueeze(3) # n * (h//4*w//4) * (c*4*4) * 1

        # cross-scale attention * 4
        index_output_s1 = index_output_s1.contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        index_output_s1 = F.unfold(index_output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)
        index_output_s1 = index_output_s1.permute(0, 3, 1, 2)
        index_output_s1 = F.normalize(index_output_s1, dim=3) # n * (h//4*w//4) * t * (c*4*4)
        matrix_index = torch.matmul(index_output_s1, curr_feat).squeeze(3) # n * (h//4*w//4) * t
        matrix_index = matrix_index.view(n,feat_num,t)# n * (h//4*w//4) * t

        corr_soft, corr_index = torch.max(matrix_index, dim=2)# n * (h//4*w//4)


        corr_soft = corr_soft.unsqueeze(1).expand(-1,feat_len,-1)
        corr_soft = F.fold(corr_soft, output_size=(int(h), int(w)), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        # Aggr
        output_s1 = output_s1.contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        output_s1 = F.unfold(output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)
        output_s1 = torch.gather(output_s1.contiguous().view(n,t,feat_len,feat_num), 1, corr_index.view(n,1,1,feat_num).expand(-1,-1,feat_len,-1))# n * 1 * (c*4*4) * (h//4*w//4)
        output_s1 = output_s1.squeeze(1)
        output_s1 = F.fold(output_s1, output_size=(int(h), int(w)), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        # Aggr
        output_s2 = output_s2.contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        output_s2 = F.unfold(output_s2, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)  
        output_s2 = torch.gather(output_s2.contiguous().view(n,t,feat_len,feat_num), 1, corr_index.view(n,1,1,feat_num).expand(-1,-1,feat_len,-1))# n * 1 * (c*4*4) * (h//4*w//4)
        output_s2 = output_s2.squeeze(1)
        output_s2 = F.fold(output_s2, output_size=(int(h), int(w)), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        # Aggr
        output_s3 = output_s3.contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        output_s3 = F.unfold(output_s3, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)  
        output_s3 = torch.gather(output_s3.contiguous().view(n,t,feat_len,feat_num), 1, corr_index.view(n,1,1,feat_num).expand(-1,-1,feat_len,-1))# n * 1 * (c*4*4) * (h//4*w//4)
        output_s3 = output_s3.squeeze(1)
        output_s3 = F.fold(output_s3, output_size=(int(h), int(w)), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        out = torch.cat([output_s1,output_s2,output_s3], dim=1)
        out = self.fusion(out)
        out = out * corr_soft
        out += anchor_feat
        return out
