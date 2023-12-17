import argparse
import cv2
import glob
import os
from tqdm import tqdm
import numpy as np
import torch
from yaml import load
import matplotlib as m
import matplotlib.pyplot as plt

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.adacode_contrast_arch import AdaCodeSRNet_Contrast
from basicsr.utils.download_util import load_file_from_url 

pretrain_model_url = {
    'SR_x2': 'https://github.com/kechunl/AdaCode/releases/download/v0-pretrain_models/AdaCode_SR_X2_model_g.pth',
    'SR_x4': 'https://github.com/kechunl/AdaCode/releases/download/v0-pretrain_models/AdaCode_SR_X4_model_g.pth',
    'inpaint': 'https://github.com/kechunl/AdaCode/releases/download/v0-pretrain_models/AdaCode_Inpaint_model_g.pth',
}

def main():
    """Inference demo for FeMaSR 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-w', '--weight', type=str, default=None, help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-t', '--task', type=str, choices=['sr', 'inpaint'], help='inference task')
    parser.add_argument('-s', '--out_scale', type=int, default=4, help='final upsampling scale of the image for SR task')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600, help='Max image size for whole image inference, otherwise use tiled_test')
    parser.add_argument('--vis_weight', action='store_true', help='visualize weight map')
    args = parser.parse_args()

    if args.vis_weight:
        os.makedirs(os.path.join(args.output, 'weight_map'), exist_ok=True)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    if args.task == 'sr':
        model_url_key = f'SR_x{args.out_scale}' 
        scale = args.out_scale
        bs = 8
    else:
        model_url_key = 'inpaint'
        scale = 1
        bs = 2

    if args.weight is None:
        os.makedirs('./checkpoint', exist_ok=True)
        weight_path = load_file_from_url(pretrain_model_url[f'{model_url_key}'], model_dir='./checkpoint')
    else:
        weight_path = args.weight
    
    # set up the model
    model_params = torch.load(weight_path)['params']
    codebook_dim = np.array([v.size() for k,v in model_params.items() if 'quantize_group' in k])
    codebook_dim_list = []
    for k in codebook_dim:
        temp = k.tolist()
        temp.insert(0,32)
        codebook_dim_list.append(temp)
    model = AdaCodeSRNet_Contrast(codebook_params=codebook_dim_list, LQ_stage=True, AdaCode_stage=True, weight_softmax=False, batch_size=bs, scale_factor=scale).to(device)
    model.load_state_dict(torch.load(weight_path)['params'], strict=False)
    model.eval()
    
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        try:
            img_name = os.path.basename(path)
            pbar.set_description(f'Test {img_name}')

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img_tensor = img2tensor(img).to(device) / 255.
            img_tensor = img_tensor.unsqueeze(0)

            max_size = args.max_size ** 2 
            h, w = img_tensor.shape[2:]
            if h * w < max_size: 
                output = model.test(img_tensor, vis_weight=args.vis_weight)
            else:
                output = model.test_tile(img_tensor, vis_weight=args.vis_weight)

            if args.vis_weight:
                weight_map = output[1]
                vis_weight(weight_map, os.path.join(args.output, 'weight_map', img_name))
                output = output[0]
            output_img = tensor2img(output)

            save_path = os.path.join(args.output, f'{img_name}')
            imwrite(output_img, save_path)
            pbar.update(1)
        except:
            print(path, ' fails.')
    pbar.close()

def vis_weight(weight, save_path):
    # weight: B x n x 1 x H x W
    weight = weight.cpu().numpy()
    # normalize weights
    # norm_weight = weight
    norm_weight = (weight - weight.mean()) / weight.std() / 2
    norm_weight = np.abs(norm_weight)
    norm_weight *= 255
    norm_weight = np.clip(norm_weight, 0, 255)
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
    main()
