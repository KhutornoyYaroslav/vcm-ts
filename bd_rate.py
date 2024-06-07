import argparse
import json
import os

import bjontegaard as bd


def str2bool(s):
    return s.lower() in ('true', '1')


def fix_curve(points, eps=1e-8):
    is_asc = points[0] < points[-1]
    result = []
    for point in points:
        if len(result) == 0:
            result.append(point)
            continue

        if is_asc:
            if point > result[-1]:
                result.append(point)
            else:
                result.append(result[-1] + eps)
        else:
            if point < result[-1]:
                result.append(point)
            else:
                result.append(result[-1] - eps)

    return result


def compute_bd(metrics, anchor, method, out_dir):
    codecs = sorted(list(metrics.keys()))
    videos = sorted(list(metrics[codecs[0]].keys()))
    out_file = os.path.join(out_dir, "bd_metrics.txt")
    if os.path.exists(out_file):
        os.remove(out_file)
    for video in videos:
        rate_anchor = [info['bpp'] for info in metrics[anchor][video]]
        psnr_anchor = [info['psnr'] for info in metrics[anchor][video]]
        detection_models = sorted(list(metrics[codecs[0]][video][0]['mean_ap'].keys()))
        map_anchors = {}
        for detection_model in detection_models:
            map_anchors[detection_model] = [info['mean_ap'][detection_model]['map'] for info in
                                            metrics[anchor][video]]
            map_anchors[detection_model] = fix_curve(map_anchors[detection_model])

        for codec in codecs:
            if codec == anchor:
                continue
            rate_test = [info['bpp'] for info in metrics[codec][video]]
            psnr_test = [info['psnr'] for info in metrics[codec][video]]

            bd_rate_psnr = bd.bd_rate(rate_anchor, psnr_anchor, rate_test, psnr_test, method=method)
            bd_psnr = bd.bd_psnr(rate_anchor, psnr_anchor, rate_test, psnr_test, method=method)

            with open(out_file, "a") as f:
                f.write(f"Codec {codec} for {video}\n")
                f.write(f"\tBD-Rate (PSNR): {bd_rate_psnr:.4f} %\n")
                f.write(f"\tBD-PSNR: {bd_psnr:.4f} dB\n")
            for detection_model in detection_models:
                map_test = [info['mean_ap'][detection_model]['map'] for info in metrics[codec][video]]
                map_test = fix_curve(map_test)
                bd_rate_map = bd.bd_rate(rate_anchor, map_anchors[detection_model], rate_test, map_test, method=method)
                bd_map = bd.bd_psnr(rate_anchor, map_anchors[detection_model], rate_test, map_test, method=method)
                with open(out_file, "a") as f:
                    f.write(f"\tBD-mAP for model {detection_model}\n")
                    f.write(f"\t\tBD-Rate (mAP): {bd_rate_map:.4f} %\n")
                    f.write(f"\t\tBD-mAP: {bd_map:.4f} %\n")


def compute_bd_gop(metrics, anchor, method, out_dir):
    codecs = sorted(list(metrics.keys()))
    videos = sorted(list(metrics[codecs[0]].keys()))
    gop_metrics = {}
    out_file = os.path.join(out_dir, "bd_metrics.txt")
    if os.path.exists(out_file):
        os.remove(out_file)
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
                map_anchors[detection_model] = fix_curve(map_anchors[detection_model])

            with open(out_file, "a") as f:
                f.write(f"Codec {codec} with anchor {anchor}:\n")
            for gop in gops:
                if gop == anchor:
                    continue
                rate_test = [info['bpp'] for info in gop_metrics[codec][gop][video]]
                psnr_test = [info['psnr'] for info in gop_metrics[codec][gop][video]]

                bd_rate_psnr = bd.bd_rate(rate_anchor, psnr_anchor, rate_test, psnr_test, method=method)
                bd_psnr = bd.bd_psnr(rate_anchor, psnr_anchor, rate_test, psnr_test, method=method)

                with open(out_file, "a") as f:
                    f.write(f"\tGOP {gop} for {video}\n")
                    f.write(f"\t\tBD-Rate (PSNR): {bd_rate_psnr:.4f} %\n")
                    f.write(f"\t\tBD-PSNR: {bd_psnr:.4f} dB\n")
                for detection_model in detection_models:
                    map_test = [info['mean_ap'][detection_model]['map'] for info in gop_metrics[codec][gop][video]]
                    map_test = fix_curve(map_test)
                    bd_rate_map = bd.bd_rate(rate_anchor, map_anchors[detection_model], rate_test, map_test,
                                             method=method)
                    bd_map = bd.bd_psnr(rate_anchor, map_anchors[detection_model], rate_test, map_test, method=method)
                    with open(out_file, "a") as f:
                        f.write(f"\t\tBD-mAP for model {detection_model}\n")
                        f.write(f"\t\t\tBD-Rate (mAP): {bd_rate_map:.4f} %\n")
                        f.write(f"\t\t\tBD-mAP: {bd_map:.4f} %\n")


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Bjøntegaard-Delta metrics calculation')
    parser.add_argument("--decod-dir", dest="decod_dir", required=False, type=str,
                        default="data/huawei/outputs/decod", help="Path to decoded dir")
    parser.add_argument("--out-path", dest="out_path", required=False, type=str,
                        default="outputs/benchmark/decod", help="Path to output dir")
    parser.add_argument("--anchor", dest="anchor", required=False, type=str,
                        default="HEVC veryslow", help="Anchor name")
    parser.add_argument("--method", dest="method", required=False, type=str,
                        default="pchip", help="Approximation method for curves")
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

    os.makedirs(args.out_path, exist_ok=True)
    if args.compare_gop:
        compute_bd_gop(metrics, args.anchor, args.method, args.out_path)
    else:
        compute_bd(metrics, args.anchor, args.method, args.out_path)

if __name__ == '__main__':
    main()
