import torch
import argparse
import argparse

import torch


def parse_weights(weights_path: str):
    weights = torch.load(weights_path, map_location=torch.device('cpu'))
    keys = []
    for key in weights:
        keys.append(key.split(".")[0])
    keys = set(keys)
    for key in sorted(keys):
        print(key)
    return weights, keys

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Test video model weights parser')
    parser.add_argument("--weights-path", dest="weights_path", required=False, type=str,
                        default="DCVC_HEM/checkpoints/acmmm2022_video_psnr.pth.tar",
                        help="Path to config file")

    args = parser.parse_args()
    weights, names = parse_weights(args.weights_path)


if __name__ == '__main__':
    main()
