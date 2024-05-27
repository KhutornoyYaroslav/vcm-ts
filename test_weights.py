import argparse

import torch


def parse_weights(weights_path: str):
    weights = torch.load(weights_path, map_location=torch.device('cpu'))

    keys = []
    for key in weights:
        keys.append(key.split(".")[0])
    keys = set(keys)

    return weights, keys


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Test video model weights parser')
    parser.add_argument("--weights-path", dest="weights_path", required=False, type=str,
                        default="pretrained/acmmm2022_video_psnr.pth",
                        help="Path to config file")

    args = parser.parse_args()
    weights, names = parse_weights(args.weights_path)

    # Show total list of parameters
    for name in sorted(names):
        print(name)

    # Show global quanization values
    for key, val in weights.items():
        if key in ['y_q_scale', 'mv_y_q_scale']:
            print(key, val.tolist())


if __name__ == '__main__':
    main()
