import torch
import torch.nn.functional as F
from torch import nn
from core.modelling.optflow.spynet import flow_warp
from core.modelling.utils.residual import ResidualBlocksWithInputConv
from core.modelling.dct.dct import dct_layer, reverse_dct_layer, remove_image_padding, resize_flow


# class FTT(nn.Module):
#     """
#     SR_imgs, (n, t, c, h, w)
#     high_frequency_imgs, (n, t, c, h, w)
#     flows, (n, t-1, 2, h, w)

#     """

#     def __init__(self, dct_kernel=(8,8), d_model=512, n_heads=8):
#         super().__init__()
#         self.dct_kernel = dct_kernel
#         self.dct = dct_layer(in_c=3, h=dct_kernel[0], w=dct_kernel[1])
#         self.rdct = reverse_dct_layer(out_c=3, h=dct_kernel[0], w=dct_kernel[1])
#         self.conv_layer1 = nn.Conv2d(3*self.dct_kernel[0]*self.dct_kernel[1], 512, 1, 1, 0, bias=True)

#         self.feat_extractor = ResidualBlocksWithInputConv(512, 512, 3)
#         self.resblocks = ResidualBlocksWithInputConv(512*2, 512, 3)
#         self.fusion = nn.Sequential(
#             nn.Conv2d(3 * 512, 512, 1, 1, 0, bias=True),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(512, 512, 1, 1, 0, bias=True)
#         )
        
#         self.ftta = FTTA_layer(channel=512, d_model=d_model, n_heads=n_heads)
#         self.conv_layer2 = nn.Conv2d(512, 3*self.dct_kernel[0]*self.dct_kernel[1], 1, 1, 0, bias=True)

#     def forward(self, bicubic_imgs, high_frequency_imgs, flows, padiings, to_cpu=False):
#         n,t,c,h,w = bicubic_imgs.shape
#         padding_h, padding_w = padiings
#         flows_forward, flows_backward = flows
#         #resize flows
#         if flows_forward is not None:
#             flows_forward = resize_flow(flows_forward.view(-1, 2, h, w), size_type='shape', sizes=(h//self.dct_kernel[0], w//self.dct_kernel[1]))
#             flows_forward = flows_forward.view(n, t-1, 2, h//self.dct_kernel[0], w//self.dct_kernel[1])
#         flows_backward = resize_flow(flows_backward.view(-1, 2, h, w), size_type='shape', sizes=(h//self.dct_kernel[0], w//self.dct_kernel[1]))
#         flows_backward = flows_backward.view(n, t-1, 2, h//self.dct_kernel[1], w//self.dct_kernel[1])

#         #to frequency domain
#         dct_bic_0 = self.dct(bicubic_imgs.view(-1, c, h, w))
#         dct_bic = F.normalize(dct_bic_0.view(n*t, c*self.dct_kernel[0]*self.dct_kernel[1], -1), dim=2).view(n*t, -1, h//self.dct_kernel[0], w//self.dct_kernel[1])
        
#         dct_hfi_0 = self.dct(high_frequency_imgs.view(-1, c, h, w))
#         dct_hfi = F.normalize(dct_hfi_0.view(n*t, c*self.dct_kernel[0]*self.dct_kernel[1], -1), dim=2).view(n*t, -1, h//self.dct_kernel[0], w//self.dct_kernel[1])
#         dct_hfi_0 = dct_hfi_0.view(n, t, -1, h//self.dct_kernel[0], w//self.dct_kernel[1])


#         dct_bic_fea = self.feat_extractor(self.conv_layer1(dct_bic)).view(n, t, 512, h//self.dct_kernel[0], w//self.dct_kernel[1])
#         dct_hfi_fea = self.feat_extractor(self.conv_layer1(dct_hfi)).view(n, t, 512, h//self.dct_kernel[0], w//self.dct_kernel[1])

#         n,t,c,h,w = dct_hfi_fea.shape


#         hfi_backward_list = []
#         hfi_prop = dct_hfi.new_zeros(n, c, h, w)
#         #backward
#         for i in range(t-1, -1, -1):
#             bic =  dct_bic_fea[:, i, :, :, :]
#             hfi = dct_hfi_fea[:, i, :, :, :]
#             if i < t-1:
#                 flow = flows_backward[:, i, :, :, :]
#                 hfi_prop = flow_warp(hfi_prop, flow.permute(0, 2, 3, 1), padding_mode='border')

#                 hfi_ = self.ftta(bic, hfi, hfi)
#                 hfi_prop = self.ftta(hfi_, hfi_prop, hfi_prop)

#             hfi_prop = torch.cat([hfi, hfi_prop], dim=1)
#             hfi_prop = self.resblocks(hfi_prop)
#             hfi_backward_list.append(hfi_prop) #(b,c,h,w)
#         #forward
#         out_fea = hfi_backward_list[::-1]

#         final_out = []
#         hfi_prop = torch.zeros_like(hfi_prop)
#         for i in range(t):
#             bic =  dct_bic_fea[:, i, :, :, :]
#             hfi = dct_hfi_fea[:, i, :, :, :]
#             if i > 0:
#                 if flows_forward is not None:
#                     flow = flows_forward[:, i - 1, :, :, :]
#                 else:
#                     flow = flows_backward[:, -i, :, :, :]
#                 # flow = flows_forward[:, i-1, :, :, :]
#                 hfi_prop = flow_warp(hfi_prop, flow.permute(0, 2, 3, 1), padding_mode='border')

#                 # hfi_prop = self.ftta(bic, hfi, hfi_prop)
#                 hfi_ = self.ftta(bic, hfi, hfi)
#                 hfi_prop = self.ftta(hfi_, hfi_prop, hfi_prop)
                
#             hfi_prop = torch.cat([hfi, hfi_prop], dim=1)
#             hfi_prop = self.resblocks(hfi_prop)
            
#             out = torch.cat([out_fea[i], hfi, hfi_prop], dim=1)
#             out = self.conv_layer2(self.fusion(out)) + dct_hfi_0[:, i, :, :, :]
#             out = self.rdct(out) + high_frequency_imgs[:, i, :, :, :]

#             out = remove_image_padding(out, padding_h, padding_w)
#             if to_cpu: 
#                 final_out.append(out.cpu())
#             else: 
#                 final_out.append(out)
#         return torch.stack(final_out, dim=1)


# class FTTA_layer(nn.Module):
#     def __init__(self, channel=192, d_model=512, n_heads=8, patch_k=(8, 8), patch_stride=8):
#         super().__init__()
#         self.patch_k = patch_k
#         self.patch_stride = patch_stride
#         inplances = (channel // 64) * patch_k[0] * patch_k[1]

#         self.layer_q = nn.Linear(inplances, d_model)
#         self.layer_k = nn.Linear(inplances, d_model)
#         self.layer_v = nn.Linear(inplances, d_model)

#         self.MultiheadAttention = nn.MultiheadAttention(d_model, n_heads)
#         self.norm1 = nn.LayerNorm(d_model)

#         self.linear1 = nn.Linear(d_model, d_model)
#         self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
#         self.norm2 = nn.LayerNorm(d_model)
#         self.linear2 = nn.Linear(d_model, inplances)

#     def forward_ffn(self, x):
#         x2 = self.linear1(x)
#         x2= self.activation(x2)
#         x = x2 + x
#         x = self.norm2(x)
#         x = self.linear2(x)

#         return x


#     def forward(self, q, k, v):
#         '''
#         q, k, v, (n, 512, h, w)
#         frequency attention
#         '''
        
#         N,C,H,W = q.shape

#         qs = q.view(N*64, -1, H, W)
#         ks = k.view(N*64, -1, H, W)
#         vs = v.view(N*64, -1, H, W)

#         qs = torch.nn.functional.unfold(qs, self.patch_k , dilation=1, padding=0, stride=self.patch_stride) #(N*64, 3*8*8, num)
#         ks = torch.nn.functional.unfold(ks, self.patch_k , dilation=1, padding=0, stride=self.patch_stride) #(N*64, 3*8*8, num)
#         vs = torch.nn.functional.unfold(vs, self.patch_k , dilation=1, padding=0, stride=self.patch_stride) #(N*64, 3*8*8, num)

#         BF, D, num = qs.shape
#         qs = qs.view(N, 64, D, num).permute(0, 1, 3, 2).reshape(N, -1, D) #(Batch, F*num, dim=3*8*8)
#         ks = ks.view(N, 64, D, num).permute(0, 1, 3, 2).reshape(N, -1, D)
#         vs = vs.view(N, 64, D, num).permute(0, 1, 3, 2).reshape(N, -1, D)

#         qs = self.layer_q(qs) #(batch, F*num, d_model)
#         ks = self.layer_k(ks)
#         vs = self.layer_v(vs)

#         qs = qs.permute(1, 0, 2) #L,N,E
#         ks = ks.permute(1, 0, 2)
#         vs = vs.permute(1, 0, 2)

#         ttn_output, attn_output_weights  = self.MultiheadAttention(qs, ks, vs)
#         out = ttn_output + vs
#         out = self.norm1(out) #LNE

#         out = out.permute(1, 0, 2) #NLE, (batch, F*num, dim=d_model)

#         out = self.forward_ffn(out) #N,L,E,
#         out = out.view(N, 64, num, D).permute(0, 1, 3, 2).reshape(-1, D, num) #(batch*64, 3*8*8, num)
#         out = torch.nn.functional.fold(out, (H,W), self.patch_k, dilation=1, padding=0, stride=self.patch_stride) #(batch*64, 3, H, W)
#         out = out.view(N, -1, H, W)

#         return out


class FTT(nn.Module):
    """
    SR_imgs, (n, t, c, h, w)
    high_frequency_imgs, (n, t, c, h, w)
    flows, (n, t-1, 2, h, w)

    """

    def __init__(self, dct_kernel=(8,8), d_model=512, n_heads=8):
        super().__init__()
        self.dct_kernel = dct_kernel
        self.dct = dct_layer(in_c=3, h=dct_kernel[0], w=dct_kernel[1])
        self.rdct = reverse_dct_layer(out_c=3, h=dct_kernel[0], w=dct_kernel[1])

        self.conv_layer1 = nn.Conv2d(192, 512, 1, 1, 0, bias=True)
        self.feat_extractor = ResidualBlocksWithInputConv(512, 512, 3)
        self.resblocks = ResidualBlocksWithInputConv(512*2, 512, 3)
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * 512, 512, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0, bias=True)
        )
        
        self.ftta = FTTA_layer(channel=512, d_model=d_model, n_heads=n_heads)

        self.conv_layer2 = nn.Conv2d(512, 192, 1, 1, 0, bias=True)

    def forward(self, bicubic_imgs, high_frequency_imgs, flows, padiings, to_cpu=False):
        n,t,c,h,w = bicubic_imgs.shape
        padding_h, padding_w = padiings
        flows_forward, flows_backward = flows
        #resize flows
        if flows_forward is not None:
            flows_forward = resize_flow(flows_forward.view(-1, 2, h, w), size_type='shape', sizes=(h//self.dct_kernel[0], w//self.dct_kernel[1]))
            flows_forward = flows_forward.view(n, t-1, 2, h//self.dct_kernel[0], w//self.dct_kernel[1])
        flows_backward = resize_flow(flows_backward.view(-1, 2, h, w), size_type='shape', sizes=(h//self.dct_kernel[0], w//self.dct_kernel[1]))
        flows_backward = flows_backward.view(n, t-1, 2, h//self.dct_kernel[1], w//self.dct_kernel[1])

        #to frequency domain
        dct_bic_0 = self.dct(bicubic_imgs.view(-1, c, h, w))
        dct_bic = F.normalize(dct_bic_0.view(n*t, c*8*8, -1), dim=2).view(n*t, -1, h//8, w//8)
        
        dct_hfi_0 = self.dct(high_frequency_imgs.view(-1, c, h, w))
        dct_hfi = F.normalize(dct_hfi_0.view(n*t, c*8*8, -1), dim=2).view(n*t, -1, h//8, w//8)
        dct_hfi_0 = dct_hfi_0.view(n, t, -1, h//self.dct_kernel[0], w//self.dct_kernel[1])


        dct_bic_fea = self.feat_extractor(self.conv_layer1(dct_bic)).view(n, t, 512, h//self.dct_kernel[0], w//self.dct_kernel[1])
        dct_hfi_fea = self.feat_extractor(self.conv_layer1(dct_hfi)).view(n, t, 512, h//self.dct_kernel[0], w//self.dct_kernel[1])

        n,t,c,h,w = dct_hfi_fea.shape


        hfi_backward_list = []
        hfi_prop = dct_hfi.new_zeros(n, c, h, w)
        #backward
        for i in range(t-1, -1, -1):
            bic =  dct_bic_fea[:, i, :, :, :]
            hfi = dct_hfi_fea[:, i, :, :, :]
            if i < t-1:
                flow = flows_backward[:, i, :, :, :]
                hfi_prop = flow_warp(hfi_prop, flow.permute(0, 2, 3, 1), padding_mode='border')

                hfi_ = self.ftta(bic, hfi, hfi)
                hfi_prop = self.ftta(hfi_, hfi_prop, hfi_prop)

            hfi_prop = torch.cat([hfi, hfi_prop], dim=1)
            hfi_prop = self.resblocks(hfi_prop)
            hfi_backward_list.append(hfi_prop) #(b,c,h,w)
        #forward
        out_fea = hfi_backward_list[::-1]

        final_out = []
        hfi_prop = torch.zeros_like(hfi_prop)
        for i in range(t):
            bic =  dct_bic_fea[:, i, :, :, :]
            hfi = dct_hfi_fea[:, i, :, :, :]
            if i > 0:
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                # flow = flows_forward[:, i-1, :, :, :]
                hfi_prop = flow_warp(hfi_prop, flow.permute(0, 2, 3, 1), padding_mode='border')

                # hfi_prop = self.ftta(bic, hfi, hfi_prop)
                hfi_ = self.ftta(bic, hfi, hfi)
                hfi_prop = self.ftta(hfi_, hfi_prop, hfi_prop)
                
            hfi_prop = torch.cat([hfi, hfi_prop], dim=1)
            hfi_prop = self.resblocks(hfi_prop)
            
            out = torch.cat([out_fea[i], hfi, hfi_prop], dim=1)
            out = self.conv_layer2(self.fusion(out)) + dct_hfi_0[:, i, :, :, :]
            out = self.rdct(out) + high_frequency_imgs[:, i, :, :, :]

            out = remove_image_padding(out, padding_h, padding_w)
            if to_cpu: 
                final_out.append(out.cpu())
            else: 
                final_out.append(out)
        return torch.stack(final_out, dim=1)

class FTT_encoder(nn.Module):
    def __init__(self, channel=192, d_model=512, n_heads=8, num_layer=3):
        super().__init__()
        self.num_layer = num_layer

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(
                FTTA_layer(channel, d_model, n_heads)
            )
    def forward(self, q, k, v):
        v = self.layers[0](q, k, v)
        for i in range(1, self.num_layer):
            v = self.layers[i](k, v, v)
        return v



class FTTA_layer(nn.Module):

    def __init__(self, channel=192, d_model=512, n_heads=8, patch_k=(8,8), patch_stride=8):
        super().__init__()
        self.patch_k = patch_k
        self.patch_stride = patch_stride
        inplances = (channel // 64) * patch_k[0] * patch_k[1]


        self.layer_q = nn.Linear(inplances, d_model)
        self.layer_k = nn.Linear(inplances, d_model)
        self.layer_v = nn.Linear(inplances, d_model)

        self.MultiheadAttention = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_model)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(d_model, inplances)

    def forward_ffn(self, x):
        x2 = self.linear1(x)
        x2= self.activation(x2)
        x = x2 + x
        x = self.norm2(x)

        x = self.linear2(x)

        return x


    def forward(self, q, k, v):
        '''
        q, k, v, (n, 512, h, w)
        frequency attention
        '''
        
        N,C,H,W = q.shape

        qs = q.view(N*64, -1, H, W)
        ks = k.view(N*64, -1, H, W)
        vs = v.view(N*64, -1, H, W)

        qs = torch.nn.functional.unfold(qs, self.patch_k , dilation=1, padding=0, stride=self.patch_stride) #(N*64, 3*8*8, num)
        ks = torch.nn.functional.unfold(ks, self.patch_k , dilation=1, padding=0, stride=self.patch_stride) #(N*64, 3*8*8, num)
        vs = torch.nn.functional.unfold(vs, self.patch_k , dilation=1, padding=0, stride=self.patch_stride) #(N*64, 3*8*8, num)

        BF, D, num = qs.shape
        qs = qs.view(N, 64, D, num).permute(0, 1, 3, 2).reshape(N, -1, D) #(Batch, F*num, dim=3*8*8)
        ks = ks.view(N, 64, D, num).permute(0, 1, 3, 2).reshape(N, -1, D)
        vs = vs.view(N, 64, D, num).permute(0, 1, 3, 2).reshape(N, -1, D)

        qs = self.layer_q(qs) #(batch, F*num, d_model)
        ks = self.layer_k(ks)
        vs = self.layer_v(vs)

        qs = qs.permute(1, 0, 2) #L,N,E
        ks = ks.permute(1, 0, 2)
        vs = vs.permute(1, 0, 2)

        ttn_output, attn_output_weights  = self.MultiheadAttention(qs, ks, vs)
        out = ttn_output + vs
        out = self.norm1(out) #LNE

        out = out.permute(1, 0, 2) #NLE, (batch, F*num, dim=d_model)

        out = self.forward_ffn(out) #N,L,E,
        out = out.view(N, 64, num, D).permute(0, 1, 3, 2).reshape(-1, D, num) #(batch*64, 3*8*8, num)
        out = torch.nn.functional.fold(out, (int(H), int(W)), self.patch_k, dilation=1, padding=0, stride=self.patch_stride) #(batch*64, 3, H, W)
        out = out.view(N, -1, H, W)

        return out