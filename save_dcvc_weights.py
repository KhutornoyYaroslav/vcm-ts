import argparse

import torch


def save_weights(weights_path: str, out_path: str):
    # load model weights
    weights = torch.load(weights_path, map_location=torch.device('cpu'))

    # separate layers
    new_state_dict = {}
    for name, param in weights['model'].items():
        new_name = name.replace("dmc.", '')  # remove `dmc.`
        new_state_dict[new_name] = param

    torch.save(new_state_dict, out_path)

    return new_state_dict


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Save trained weights in pth file for DCVC-HEM model')
    parser.add_argument("--weights-path", dest="weights_path", required=False, type=str,
                        default="outputs/full_train/model_final.pth",
                        help="Path to model weights file")
    parser.add_argument("--out-path", dest="out_path", required=False, type=str,
                        default="pretrained/ours_video_psnr.pth",
                        help="Path to out DCVC-HEM weights file")

    args = parser.parse_args()
    dcvc_weights = save_weights(args.weights_path, args.out_path)

    # Show total list of spynet parameters
    print('Save DCVC-HEM weights in ' + args.out_path)
    for name in sorted(dcvc_weights):
        print(name)


if __name__ == '__main__':
    main()
