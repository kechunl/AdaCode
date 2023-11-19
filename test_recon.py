import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load
import pdb
import numpy as np
import matplotlib as m
import matplotlib.pyplot as plt
from PIL import Image
import pyiqa

from basicsr.utils import img2tensor, tensor2img, imwrite 
from basicsr.archs.adacode_arch import AdaCodeSRNet
from basicsr.archs.femasr_arch import FeMaSRNet 
from basicsr.archs.adacode_contrast_arch import AdaCodeSRNet_Contrast
from basicsr.utils.download_util import load_file_from_url 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#eval metrics
metric_funcs = {}
metric_funcs['psnr'] = pyiqa.create_metric('psnr', device=device, crop_border=4, test_y_channel=True)
metric_funcs['ssim'] = pyiqa.create_metric('ssim', device=device, crop_border=4, test_y_channel=True)
metric_funcs['lpips'] = pyiqa.create_metric('lpips', device=device)

def main(args):
    """Inference demo for FeMaSR 
    """
    metric_results = {'psnr': 0, 'ssim': 0, 'lpips': 0}

    weight_path = args.weight
    
    # set up the model
    model_params = torch.load(weight_path)['params']
    codebook_dim = np.array([v.size() for k,v in model_params.items() if 'quantize_group' in k])
    codebook_dim_list = []
    for k in codebook_dim:
        temp = k.tolist()
        temp.insert(0,32)
        codebook_dim_list.append(temp)
    # recon_model = FeMaSRNet(codebook_params=[[32, 512, 256]], LQ_stage=False, scale_factor=2).to(device)
    recon_model = AdaCodeSRNet_Contrast(codebook_params=codebook_dim_list, LQ_stage=False, AdaCode_stage=True, batch_size=2, weight_softmax=False).to(device)
    recon_model.load_state_dict(model_params, strict=False)
    recon_model.eval()
    
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*.png')))

    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        try:
            img_name = os.path.basename(path)
            pbar.set_description(f'Test {img_name}')

            # recon
            img_HR = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img_HR_tensor = img2tensor(img_HR).to(device) / 255.
            img_HR_tensor = img_HR_tensor.unsqueeze(0)

            max_size = args.max_size ** 2 
            h, w = img_HR_tensor.shape[2:]
            if h * w < max_size: 
                output_HR = recon_model.test(img_HR_tensor)
            else:
                output_HR = recon_model.test_tile(img_HR_tensor)

            output = output_HR[0]
            output_img = tensor2img(output)

            save_path = os.path.join(args.output, f'{img_name}')
            imwrite(output_img, save_path)

            for name in metric_funcs.keys():
                metric_results[name] += metric_funcs[name](img_HR_tensor, output.unsqueeze(0)).item()
            pbar.update(1)
        except:
            print(path, ' fails.')
    pbar.close()

    for name in metric_results.keys():
        metric_results[name] /= len(paths)
    print('Result for {}'.format(args.weight))
    print(metric_results)


def vis_weight(weight, save_path):
    # weight: B x n x 1 x H x W
    weight = weight.cpu().numpy()
    # normalize weights
    # norm_weight = (weight - weight.mean()) / weight.std() / 2
    # norm_weight = np.abs(norm_weight)
    norm_weight = weight
    norm_weight *= 255
    norm_weight = np.clip(norm_weight, 0, 255)
    # norm_weight += 127
    norm_weight = norm_weight.astype(np.uint8)
    # visualize
    display_grid = np.zeros((weight.shape[3], (weight.shape[4]+1)*weight.shape[1]-1))
    for img_id in range(len(norm_weight)):
        for c in range(norm_weight.shape[1]):
            display_grid[:, c*weight.shape[4]+c:(c+1)*weight.shape[4]+c] = norm_weight[img_id, c, 0, :, :]

            # weight_path = save_path.split('.')[0] + '_{}.png'.format(str(c))
            # Image.fromarray(norm_weight[img_id, c, 0, :, :]).save(weight_path)
    plt.figure(figsize=(30,150))
    plt.axis('off')
    plt.imshow(display_grid)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-w', '--weight', type=str, default=None, help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600000, help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    main(args)
