import copy

import torch
import argparse
from DCVC_HEM.src.models.video_model import DMC


def save_weights(weights_path: str, out_path: str):
    no_weights_model = DMC()
    weights_model = copy.deepcopy(no_weights_model)

    # load model weights
    weights = torch.load(weights_path, map_location=torch.device('cpu'))

    # separate optic_flow layers
    new_state_dict = {}
    for name, param in weights.items():
        if 'optic_flow' in name:
            new_state_dict[name] = param

    torch.save(new_state_dict, out_path)

    # load only spynet weights in model
    spynet_weights = torch.load(out_path, map_location=torch.device('cpu'))
    weights_model.load_state_dict(spynet_weights, strict=False)

    # check that weights for spynet was loaded (soft)
    old_params = {}
    for name, param in no_weights_model.named_parameters():
        old_params[name] = param
    for name, param in weights_model.named_parameters():
        if 'optic_flow' in name:
            assert torch.any(torch.not_equal(old_params[name], param))
        else:
            assert torch.any(torch.eq(old_params[name], param))

    return new_state_dict


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Save SpyNet weights in pth file from DCVC-HEM model')
    parser.add_argument("--weights-path", dest="weights_path", required=False, type=str,
                        default="pretrained/acmmm2022_video_psnr.pth",
                        help="Path to model weights file")
    parser.add_argument("--out-path", dest="out_path", required=False, type=str,
                        default="pretrained/spynet.pth",
                        help="Path to out spynet weights file")

    args = parser.parse_args()
    spynet_weights = save_weights(args.weights_path, args.out_path)

    # Show total list of spynet parameters
    print('Save SpyNet weights in ' + args.out_path)
    for name in sorted(spynet_weights):
        print(name)


if __name__ == '__main__':
    main()
