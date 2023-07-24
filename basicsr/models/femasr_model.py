from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import pdb

import torch
import torchvision.utils as tvu

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import copy
from basicsr.utils import get_root_logger

import pyiqa


@MODEL_REGISTRY.register()
class FeMaSRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.AdaCode_stage = self.opt['network_g'].get('AdaCode_stage', False)
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False) 

        # obtain codebook size from pretrained codebook
        if self.AdaCode_stage:
            if not self.LQ_stage:
                load_path = self.opt['path'].get('pretrain_codebook', None)
            else:
                load_path = self.opt['path'].get('pretrain_network_hq', None)
            assert load_path is not None, 'Need to specify pretrained codebook paths in Adaptive Codebook stage'
            codebook_params = self._get_codebook_config(load_path)
            self.opt['network_g']['codebook_params'] = codebook_params

        # define network
        self.net_g = build_network(opt['network_g'])

        # define metric functions 
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items(): 
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # load pre-trained multiple codebooks, frozen codebook
        if self.AdaCode_stage and not self.LQ_stage:
            load_paths = self.opt['path'].get('pretrain_codebook', None)
            for i, load_path in enumerate(load_paths):
                self.load_codebook(self.net_g.quantize_group[i], load_path, False)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break
        # load pre-trained HQ ckpt, frozen decoder and codebook 
        elif self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

            hq_opt = self.opt['network_g'].copy()
            # hq_opt['type'] = 'AdaCodeSRNet'
            hq_opt['LQ_stage'] = False
            self.net_hq = build_network(hq_opt)
            self.net_hq = self.model_to_device(self.net_hq)
            self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])

            # net_hq doesn't require grads
            for p in self.net_hq.parameters():
                p.requires_grad = False

            self.load_network(self.net_g, load_path, False)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])
            
        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0) 
            self.net_d_best = copy.deepcopy(self.net_d)
        
        self.net_g = self.model_to_device(self.net_g)
        self.net_g_best = copy.deepcopy(self.net_g)

    def load_codebook(self, net, load_path, strict=True, param_key='params'):
        """Load codebook.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        net.state_dict()['embedding.weight'].copy_(load_net['quantize_group.0.embedding.weight'])

    def _get_codebook_config(self, load_paths):
        codebook_params = []
        if self.AdaCode_stage and not self.LQ_stage:
            for load_path in load_paths:
                load_dict = torch.load(load_path)
                assert isinstance(load_dict, dict), 'pretrain_codebook must be a dict'
                # TODO: codebook_scale 32 is hardcoded for now. change it to configurable later if needed.
                codebook_params.append([32, load_dict['params']['quantize_group.0.embedding.weight'].shape[0], load_dict['params']['quantize_group.0.embedding.weight'].shape[1]])
        elif self.AdaCode_stage and self.LQ_stage:
            load_dict = torch.load(load_paths)
            assert isinstance(load_dict, dict), 'pretrain_network_hq must be a dict'
            codebook_keys = [v for v in load_dict['params'].keys() if 'quantize_group' in v]
            for codebook_key in codebook_keys:
                codebook_params.append([32, load_dict['params'][codebook_key].shape[0], load_dict['params'][codebook_key].shape[1]])
        return codebook_params

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        # print(load_path)
        if load_path is not None:
            logger.info(f'Loading net_d from {load_path}')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
            
        self.net_d.train()
    
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter, finetune_all=False):
        train_opt = self.opt['train']

        # Finetune all parameters.
        if finetune_all:
            for p in self.net_g.parameters():
                p.requires_grad = True

        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        if self.LQ_stage and not finetune_all:
            with torch.no_grad():
                self.gt_rec, _, gt_aux = self.net_hq(self.gt)

            if train_opt.get('synfeatmatch', False):
                self.output, losses, _ = self.net_g(self.lq, gt_aux, self.net_hq.module.multiscale_encoder)
            else:
                self.output, losses, _ = self.net_g(self.lq, gt_aux)

        elif self.LQ_stage and finetune_all:
            self.output, losses, _ = self.net_g(self.lq) 

        else:
            self.output, losses, _ = self.net_g(self.gt) 

        l_g_total = 0
        loss_dict = OrderedDict()

        # ===================================================
        # codebook loss
        if train_opt.get('codebook_opt', None) and 'codebook_loss' in losses.keys():
            l_codebook = losses['codebook_loss'] * train_opt['codebook_opt']['loss_weight'] 
            l_g_total += l_codebook.mean()
            loss_dict['l_codebook'] = l_codebook.mean()

        # semantic cluster loss, only for LQ stage!
        if train_opt.get('semantic_opt', None) and 'semantic_loss' in losses.keys():
            l_semantic = losses['semantic_loss'] * train_opt['semantic_opt']['loss_weight'] 
            l_semantic = l_semantic.mean()
            l_g_total += l_semantic
            loss_dict['l_semantic'] = l_semantic

        # weight prediction loss
        if train_opt.get('weight_opt', None) and 'weight_loss' in losses.keys():
            l_weight = losses['weight_loss'] * train_opt['weight_opt']['loss_weight']
            l_weight = l_weight.mean()
            l_g_total += l_weight
            loss_dict['l_weight'] = l_weight

        # code prediction loss
        if train_opt.get('code_pred_opt', None) and 'code_pred_loss' in losses.keys():
            l_code_pred = losses['code_pred_loss'] * train_opt['code_pred_opt']['loss_weight']
            l_code_pred = l_code_pred.mean()
            l_g_total += l_code_pred
            loss_dict['l_code_pred'] = l_code_pred

        # before quant loss
        if train_opt.get('before_quant_opt', None) and 'before_quant_loss' in losses.keys():
            l_before_quant = losses['before_quant_loss'] * train_opt['before_quant_opt']['loss_weight']
            l_before_quant = l_before_quant.mean()
            l_g_total += l_before_quant
            loss_dict['l_before_quant'] = l_before_quant

        # after quant loss
        if train_opt.get('after_quant_opt', None) and 'after_quant_loss' in losses.keys():
            l_after_quant = losses['after_quant_loss'] * train_opt['after_quant_opt']['loss_weight']
            l_after_quant = l_after_quant.mean()
            l_g_total += l_after_quant
            loss_dict['l_after_quant'] = l_after_quant

        # contrastive loss
        if train_opt.get('contrast_opt', None) and 'contrast_loss' in losses.keys():
            l_contrast = losses['contrast_loss'] * train_opt['contrast_opt']['loss_weight']
            l_contrast = l_contrast.mean()
            l_g_total += l_contrast
            loss_dict['l_contrast'] = l_contrast

        # syn_feat_loss
        if train_opt.get('synfeat_opt', None) and 'syn_feat_loss' in losses.keys():
            l_syn_feat = losses['syn_feat_loss'] * train_opt['synfeat_opt']['loss_weight']
            l_syn_feat = l_syn_feat.mean()
            l_g_total += l_syn_feat
            loss_dict['l_syn_feat'] = l_syn_feat

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style
        
        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        l_g_total.mean().backward()
        self.optimizer_g.step()

        # optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000 # use smaller min_size with limited GPU memory
        lq_input = self.lq
        _, _, h, w = lq_input.shape
        if h*w < min_size:
            self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)
        self.net_g.train()
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, save_as_dir):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            
        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric') 

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            
            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}', 
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_result = self.metric_funcs[name](*metric_data)
                    self.metric_results[name] += tmp_result.item() 

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()
            
        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            
            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric, self.metric_results[self.key_metric], current_iter)
            
                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')
            else:
                # update each metric separately 
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated 
                if sum(updated): 
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    
    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx) 
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]
        if not self.LQ_stage and not self.AdaCode_stage:
            out_dict['codebook'] = self.vis_single_code()
        if hasattr(self, 'gt_rec'):
            out_dict['gt_rec'] = self.gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
