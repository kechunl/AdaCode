import pyiqa
import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
import yaml


def main():
    """Evaluation. Metrics: PSNR, SSIM, LPIPS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', type=str, help='Predicted image or folder')
    parser.add_argument('-g', '--gt', type=str, help='groundtruth image or folder')
    parser.add_argument('-o', '--output', type=str, help='Output folder')
    args = parser.parse_args()

    if args.output is None:
    	args.output = args.pred
    os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric_funcs = {}
    metric_funcs['psnr'] = pyiqa.create_metric('psnr', device=device, crop_border=4, test_y_channel=True)
    metric_funcs['ssim'] = pyiqa.create_metric('ssim', device=device, crop_border=4, test_y_channel=True)
    metric_funcs['lpips'] = pyiqa.create_metric('lpips', device=device)

    metric_results = {'psnr': 0, 'ssim': 0, 'lpips': 0}

    # image
    pred_img = glob.glob(os.path.join(args.pred, '*.png'))
    gt_img = glob.glob(os.path.join(args.gt, '*.png'))
    basename = list(set([os.path.basename(img) for img in pred_img]) & set([os.path.basename(img) for img in gt_img]))
    data = [[os.path.join(args.pred, bn), os.path.join(args.gt, bn)] for bn in basename]

    # evaluate
    for idx in range(len(data)):
    	for name in metric_funcs.keys():
    		metric_results[name] += metric_funcs[name](*data[idx]).item()

    for name in metric_results.keys():
    	metric_results[name] /= len(data)

    print(metric_results)
    with open(os.path.join(args.output, 'result.yaml'), 'w') as outfile:
    	yaml.dump(metric_results, outfile, default_flow_style=False)

if __name__ == '__main__':
    main()