import argparse
import json
import os

import bjontegaard as bd


def str2bool(s):
    return s.lower() in ('true', '1')


def compute_bd(metrics, anchor):
    codecs = sorted(list(metrics.keys()))
    videos = sorted(list(metrics[codecs[0]].keys()))
    for video in videos:
        rate_anchor = [info['bpp'] for info in metrics[anchor][video]]
        psnr_anchor = [info['psnr'] for info in metrics[anchor][video]]
        detection_models = sorted(list(metrics[codecs[0]][video][0]['mean_ap'].keys()))
        map_anchors = {}
        for detection_model in detection_models:
            map_anchors[detection_model] = [info['mean_ap'][detection_model]['map'] for info in
                                            metrics[anchor][video]]

        for codec in codecs:
            if codec == anchor:
                continue
            rate_test = [info['bpp'] for info in metrics[codec][video]]
            psnr_test = [info['psnr'] for info in metrics[codec][video]]

            bd_rate = bd.bd_rate(rate_anchor, psnr_anchor, rate_test, psnr_test, method='akima')
            bd_psnr = bd.bd_psnr(rate_anchor, psnr_anchor, rate_test, psnr_test, method='akima')

            print(f"Codec {codec} for {video}")
            print(f"\tBD-Rate: {bd_rate:.4f} %")
            print(f"\tBD-PSNR: {bd_psnr:.4f} dB")

            print(f"\tBD-mAP for models")
            for detection_model in detection_models:
                map_test = [info['mean_ap'][detection_model]['map'] for info in metrics[codec][video]]
                bd_map = bd.bd_psnr(rate_anchor, map_anchors[detection_model], rate_test, map_test, method='akima')
                print(f"\t\t{detection_model}: {bd_map:.4f} %")


def compute_bd_gop(metrics, anchor):
    codecs = sorted(list(metrics.keys()))
    videos = sorted(list(metrics[codecs[0]].keys()))
    gop_metrics = {}
    for codec in codecs:
        codec_unique_name = codec.split('gop')[0].strip()
        if codec_unique_name not in gop_metrics.keys():
            gop_metrics[codec_unique_name] = {}
        gop = str(metrics[codec][videos[0]][0]['gop'])
        gop_metrics[codec_unique_name][gop] = {}
        for video in videos:
            gop_metrics[codec_unique_name][gop][video] = metrics[codec][video]

    codec_unique_names = sorted(gop_metrics.keys())
    gops = list(map(str, sorted(list(map(int, gop_metrics[codec_unique_names[0]])))))

    for video in videos:
        for codec in codec_unique_names:
            rate_anchor = [info['bpp'] for info in gop_metrics[codec][anchor][video]]
            psnr_anchor = [info['psnr'] for info in gop_metrics[codec][anchor][video]]
            detection_models = sorted(list(gop_metrics[codec_unique_names[0]][gops[0]][video][0]['mean_ap'].keys()))
            map_anchors = {}
            for detection_model in detection_models:
                map_anchors[detection_model] = [info['mean_ap'][detection_model]['map'] for info in
                                                gop_metrics[codec][anchor][video]]

            print(f"Codec {codec} with anchor {anchor}:")
            for gop in gops:
                if gop == anchor:
                    continue
                rate_test = [info['bpp'] for info in gop_metrics[codec][gop][video]]
                psnr_test = [info['psnr'] for info in gop_metrics[codec][gop][video]]

                bd_rate = bd.bd_rate(rate_anchor, psnr_anchor, rate_test, psnr_test, method='akima')
                bd_psnr = bd.bd_psnr(rate_anchor, psnr_anchor, rate_test, psnr_test, method='akima')

                print(f"\tGOP {gop} for {video}")
                print(f"\t\tBD-Rate: {bd_rate:.4f} %")
                print(f"\t\tBD-PSNR: {bd_psnr:.4f} dB")

                print(f"\t\tBD-mAP for models")
                for detection_model in detection_models:
                    map_test = [info['mean_ap'][detection_model]['map'] for info in gop_metrics[codec][gop][video]]
                    bd_map = bd.bd_psnr(rate_anchor, map_anchors[detection_model], rate_test, map_test, method='akima')
                    print(f"\t\t\t{detection_model}: {bd_map:.4f} %")


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Bjøntegaard-Delta metrics calculation')
    parser.add_argument("--decod-dir", dest="decod_dir", required=False, type=str,
                        default="data/huawei/outputs/cut_decod_hevc", help="Path to decoded dir")
    parser.add_argument("--anchor", dest="anchor", required=False, type=str,
                        default="HEVC veryslow", help="Anchor name")
    parser.add_argument("--compare-gop", dest="compare_gop", required=False, type=str2bool,
                        default=False, help="Compare metrics for GOP")

    args = parser.parse_args()

    metrics = {}
    model_folders = [f for f in os.scandir(args.decod_dir) if f.is_dir()]
    for codec_folder in model_folders:
        metrics[codec_folder.name] = {}
        video_folders = [f for f in os.scandir(codec_folder) if f.is_dir()]
        for video_folder in video_folders:
            metrics[codec_folder.name][video_folder.name] = []
            qualities = [f for f in os.scandir(video_folder) if f.is_dir()]
            qualities.sort(key=lambda folder: folder.name)
            for quality in qualities:
                metrics_json = quality.path + '_metrics.json'
                if os.path.exists(metrics_json):
                    with open(metrics_json) as f:
                        metrics_info = json.load(f)
                    metrics[codec_folder.name][video_folder.name].append(metrics_info)
                    print(f'\t\tRead metrics for {quality.name} from json')
                else:
                    raise RuntimeError(f"No file with metrics for {quality}")

    if args.compare_gop:
        compute_bd_gop(metrics, args.anchor)
    else:
        compute_bd(metrics, args.anchor)

if __name__ == '__main__':
    main()
