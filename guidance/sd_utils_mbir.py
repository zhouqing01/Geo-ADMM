from audioop import mul
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, \
                      EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, ControlNetModel, \
                      DDIMInverseScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
import os
import random

import torchvision.transforms as T
# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator
import cv2

from .sd_step import *

def rgb2sat(img, T=None):
    max_ = torch.max(img, dim=1, keepdim=True).values + 1e-5
    min_ = torch.min(img, dim=1, keepdim=True).values
    sat = (max_ - min_) / max_
    if T is not None:
        sat = (1 - T) * sat
    return sat

def _gaussian_kernel1d(kernel_size: int, sigma: float, device, dtype):
    half = (kernel_size - 1) / 2.0
    x = torch.arange(kernel_size, device=device, dtype=dtype) - half
    w = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    w = w / (w.sum() + 1e-8)
    return w  # [K]

def gaussian_blur2d(x, kernel_size=5, sigma=1.0):
    # x: [B,C,H,W]
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    k1 = _gaussian_kernel1d(kernel_size, sigma, device, dtype)
    # separable conv
    kx = k1.view(1, 1, 1, kernel_size).repeat(C, 1, 1, 1)
    ky = k1.view(1, 1, kernel_size, 1).repeat(C, 1, 1, 1)
    x = F.conv2d(x, kx, padding=(0, kernel_size//2), groups=C)
    x = F.conv2d(x, ky, padding=(kernel_size//2, 0), groups=C)
    return x

def decompose_latent_frequency_torch(latents, kernel_size=5, sigma=1.0):
    low = gaussian_blur2d(latents, kernel_size=kernel_size, sigma=sigma)
    high = latents - low
    return low, high


def depth_to_normal(depth_map):

    B, C, H, W = depth_map.shape
    
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            device=depth_map.device, dtype=depth_map.dtype).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            device=depth_map.device, dtype=depth_map.dtype).view(1, 1, 3, 3)
    
    dx = F.conv2d(depth_map, kernel_x, padding=1)
    dy = F.conv2d(depth_map, kernel_y, padding=1)

    normal = torch.cat((-dx, -dy, torch.ones_like(depth_map)), dim=1)
    normal = F.normalize(normal, dim=1)
    
    return normal

def depth_to_normal_colored(depth_map, alpha_mask=None):

    if depth_map.dim() == 3:
        depth_map = depth_map.unsqueeze(0)
    if alpha_mask is not None and alpha_mask.dim() == 3:
        alpha_mask = alpha_mask.unsqueeze(0)
    
    B, C, H, W = depth_map.shape
    device, dtype = depth_map.device, depth_map.dtype
    
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    
    kernel_x = torch.tensor([[-1, 0, 1], 
                              [-2, 0, 2], 
                              [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], 
                              [ 0,  0,  0], 
                              [ 1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    
    dx = F.conv2d(F.pad(depth_norm, (1,1,1,1), mode='replicate'), kernel_x)
    dy = F.conv2d(F.pad(depth_norm, (1,1,1,1), mode='replicate'), kernel_y)

    gradient_scale = 2.0 
    dx = dx * gradient_scale
    dy = dy * gradient_scale
    
    normal_x = -dx
    normal_y = -dy
    normal_z = torch.ones_like(dx)
    
    normal = torch.cat([normal_x, normal_y, normal_z], dim=1)

    normal = F.normalize(normal, dim=1, eps=1e-6)

    normal_rgb = (normal + 1.0) * 0.5

    if alpha_mask is not None:

        alpha_binary = (alpha_mask > 0.5).float()
        normal_rgb = normal_rgb * alpha_binary
    
    return normal_rgb.clamp(0, 1)


def get_clean_normal(pred_normal):

    normal_np = pred_normal.detach().permute(0, 2, 3, 1).cpu().numpy()

    normal_np = ((normal_np + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    
    clean_np = np.zeros_like(normal_np)
    for b in range(normal_np.shape[0]):
        clean_np[b] = cv2.bilateralFilter(normal_np[b], d=5, sigmaColor=75, sigmaSpace=75)
    
    clean_normal = torch.from_numpy(clean_np).to(pred_normal.device).float() / 255.0
    clean_normal = (clean_normal * 2) - 1 
    return clean_normal.permute(0, 3, 1, 2)


def normal_to_rgb(normal):

    rgb = (normal + 1) * 0.5
    return rgb.clamp(0, 1)

def normal_to_grayscale(normal):

    z_component = normal[:, 2:3, :, :]  # [B, 1, H, W]
    
    gray = (z_component + 1) * 0.5
    
    return gray.clamp(0, 1)


def visualize_tensor_as_grayscale(tensor, normalize=True):
    if tensor.shape[1] > 1:
        gray = tensor.mean(dim=1, keepdim=True)
    else:
        gray = tensor
    
    if normalize:
        min_val = gray.min()
        max_val = gray.max()
        gray = (gray - min_val) / (max_val - min_val + 1e-8)
    
    return gray.clamp(0, 1)


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98], max_t_range=0.98, num_train_timesteps=None, 
                 ddim_inv=False, use_control_net=False, textual_inversion_path = None, 
                 LoRA_path = None, guidance_opt=None):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        print(f'[INFO] loading stable diffusion...')

        model_key = guidance_opt.model_key
        assert model_key is not None

        is_safe_tensor = guidance_opt.is_safe_tensor
        base_model_key = "stabilityai/stable-diffusion-v1-5" if guidance_opt.base_model_key is None else guidance_opt.base_model_key # for finetuned model only

        if is_safe_tensor:
            pipe = StableDiffusionPipeline.from_single_file(model_key, use_safetensors=True, torch_dtype=self.precision_t, load_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        self.ism = not guidance_opt.sds
        self.scheduler = DDIMScheduler.from_pretrained(model_key if not is_safe_tensor else base_model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.sche_func = ddim_step

        if use_control_net:
            controlnet_model_key = guidance_opt.controlnet_model_key
            self.controlnet_depth = ControlNetModel.from_pretrained(controlnet_model_key,torch_dtype=self.precision_t).to(device)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        pipe.enable_xformers_memory_efficient_attention()

        pipe = pipe.to(self.device)
        if textual_inversion_path is not None:
            pipe.load_textual_inversion(textual_inversion_path)
            print("load textual inversion in:.{}".format(textual_inversion_path))
        
        if LoRA_path is not None:
            from lora_diffusion import tune_lora_scale, patch_pipe
            print("load lora in:.{}".format(LoRA_path))
            patch_pipe(
                pipe,
                LoRA_path,
                patch_text=True,
                patch_ti=True,
                patch_unet=True,
            )
            tune_lora_scale(pipe.unet, 1.00)
            tune_lora_scale(pipe.text_encoder, 1.00)

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        
        self.num_train_timesteps = num_train_timesteps if num_train_timesteps is not None else self.scheduler.config.num_train_timesteps        
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(self.num_train_timesteps*(max_t_range-t_range[1]))

        self.noise_temp = None
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.noise_seed)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.rgb_latent_factors = torch.tensor([
                    # R       G       B
                    [ 0.298,  0.207,  0.208],
                    [ 0.187,  0.286,  0.173],
                    [-0.158,  0.189,  0.264],
                    [-0.184, -0.271, -0.473]
                ], device=self.device)
        

        print(f'[INFO] loaded stable diffusion!')

    def augmentation(self, *tensors):
        augs = T.Compose([
                        T.RandomHorizontalFlip(p=0.5),
                    ])
        
        channels = [ten.shape[1] for ten in tensors]
        tensors_concat = torch.concat(tensors, dim=1)
        tensors_concat = augs(tensors_concat)

        results = []
        cur_c = 0
        for i in range(len(channels)):
            results.append(tensors_concat[:, cur_c:cur_c + channels[i], ...])
            cur_c += channels[i]
        return (ten for ten in results)

    def add_noise_with_cfg(self, latents, noise, 
                           ind_t, ind_prev_t, 
                           text_embeddings=None, cfg=1.0, 
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0):

        text_embeddings = text_embeddings.to(self.precision_t)
        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
            
            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                unet_output = unet(latent_model_input, timestep_model_input, 
                                encoder_hidden_states=text_embeddings).sample
                
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond) # reverse cfg to enhance the distillation
            else:
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)
                unet_output = unet(cur_noisy_lat_, timestep_model_input, 
                                    encoder_hidden_states=uncond_text_embedding).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t-cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t-cur_ind_t

            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]


    @torch.no_grad()
    def get_text_embeds(self, prompt, resolution=(512, 512)):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                           grad_scale=1,use_control_net=False,
                           save_folder:Path=None, iteration=0, warm_up_rate = 0, weights = 0, 
                           resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None, override_latents=None, admm_external_force=None):


        # flip aug
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if override_latents is not None:
            latents = override_latents
        else:
            if as_latent:      
                latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
            else:
                latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        
        weights = weights.reshape(-1)
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])

        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1]) # make it k+1, c * t, ...

        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + np.ceil((warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t)))
        else:
            current_delta_t =  guidance_opt.delta_t

        ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            # step unroll via ddim inversion
            if not self.ism:
                prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                target = noise
            else:
                # Step 1: sample x_s with larger steps
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t, starting_ind, inverse_text_embeddings, 
                                                                                guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=guidance_opt.xs_eta)
                # Step 2: sample x_t
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t, inverse_text_embeddings, 
                                                                           guidance_opt.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True)        

                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]


        with torch.no_grad():
            latent_model_input = latents_noisy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
            if use_control_net:
                pred_depth_input = pred_depth_input[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, 512, 512).half()
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    latent_model_input,
                    tt,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=pred_depth_input,
                    return_dict=False,
                )
                unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings,
                                    down_block_additional_residuals=down_block_res_samples,
                                    mid_block_additional_residual=mid_block_res_sample).sample
            else:
                unet_output = self.unet(latent_model_input.to(self.precision_t), tt.to(self.precision_t), encoder_hidden_states=text_embeddings.to(self.precision_t)).sample

            unet_output = unet_output.reshape(1 + K, -1, 4, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            delta_DSD = weighted_perpendicular_aggregator(delta_noise_preds,\
                                                            weights,\
                                                            B)     

        pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)

        grad = w(self.alphas[t]) * (pred_noise - target)
        
        grad_low, grad_high = self.decompose_latent_frequency(grad, kernel_size=5, opt_params=guidance_opt)
        # grad_low, grad_high = decompose_latent_frequency_torch(grad, kernel_size=5, sigma=1.0)
        
        grad_high = grad_high * 2.0
        
        current_t_val = t.item()

        if current_t_val > 500:
            sds_low_freq_scale = 0.1 
        elif current_t_val > 200:
            sds_low_freq_scale = 0.3
        else:
            sds_low_freq_scale = 0.6 
        
        grad_modified = grad_high + grad_low * sds_low_freq_scale
        
        grad_norm = torch.norm(grad_modified, dim=1, keepdim=True)
        grad_modified = grad_modified * torch.minimum(torch.ones_like(grad_norm), max_grad_norm / (grad_norm + 1e-8)) # * grad_scale
        grad_modified = torch.nan_to_num(grad_scale * grad_modified)

        loss = SpecifyGradient.apply(latents, grad_modified)


        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD    
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,prev_t.item()))
            with torch.no_grad():
                d_raw = pred_depth.detach()
                a_raw = pred_alpha.detach()
                max_a = a_raw.max()
                thresh = max_a * 0.5
                mask = a_raw > thresh
                
                if mask.any():

                    obj_d = d_raw[mask]
                    v_min = torch.quantile(obj_d, 0.05)
                    v_max = torch.quantile(obj_d, 0.95)
                    enhanced_depth = (v_max - d_raw) / (v_max - v_min + 1e-7)
                    enhanced_depth = torch.clamp(enhanced_depth, 0.0, 1.0)
                    alpha_s = torch.clamp((a_raw - thresh) / (max_a - thresh + 1e-7), 0, 1)
                    enhanced_depth = enhanced_depth * alpha_s
                    enhanced_depth[~mask] = 0
                else:
                    enhanced_depth = torch.zeros_like(d_raw)
                
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)    
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))

                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

                viz_images = torch.cat([pred_rgb, 
                                        enhanced_depth.repeat(1, 3, 1, 1),
                                        pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), 
                                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        latents_rgb, latents_sp_rgb, 
                                        norm_grad,
                                        pred_x0_sp, pred_x0_pos],dim=0) 
                save_image(viz_images, save_path_iter)
                
                
        with torch.no_grad():
            raw_grad_norm = grad.norm().item()
            low_freq_norm = grad_low.norm().item()     
            high_freq_norm = grad_high.norm().item()   
            low_freq_ratio = low_freq_norm / (raw_grad_norm + 1e-8)

        monitor_dict = {
            "sds_norm": raw_grad_norm,
            "sds_final": grad_modified.norm().item(),
            "sds_low": low_freq_norm,
            "sds_high": high_freq_norm,
            "low_ratio": low_freq_ratio,
            "timestep": t.item()
        }
        
        raw_sds_scaled = grad * grad_scale
        correction = grad_modified - raw_sds_scaled
        control_variate = raw_sds_scaled - grad_modified
        with torch.no_grad():
            raw_grad_norm = grad.norm().item()
            low_freq_norm = grad_low.norm().item()
            high_freq_norm = grad_high.norm().item()
            low_freq_ratio = low_freq_norm / (raw_grad_norm + 1e-8)
            
            if admm_external_force is not None:
                admm_control_std = admm_external_force.std().item()
            else:
                admm_control_std = 0.0

            monitor_dict = {
                "sds_norm": raw_grad_norm,
                "sds_final": grad_modified.norm().item(),
                "sds_low": low_freq_norm,
                "sds_high": high_freq_norm,
                "low_ratio": low_freq_ratio,
                "timestep": t.item(),
                "score_std": raw_sds_scaled.std().item(),
                "control_std": control_variate.std().item(),
                "admm_control_std": admm_control_std,
                "score_mean": raw_sds_scaled.mean().item(),
                "control_mean": correction.mean().item()
            }

        return loss, monitor_dict


    def train_step(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                    grad_scale=1,use_control_net=False,
                    save_folder:Path=None, iteration=0, warm_up_rate = 0,
                    resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None, override_latents=None, admm_external_force=None):

        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1
        
        if override_latents is not None:
            latents = override_latents
        else:
            if as_latent:      
                latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
            else:
                latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))

        if self.noise_temp is None:
            self.noise_temp = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
        
        if guidance_opt.fix_noise:
            noise = self.noise_temp
        else:
            noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        text_embeddings = text_embeddings[:, :, ...]
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1]) # make it k+1, c * t, ...

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])

        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + (warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t))
        else:
            current_delta_t =  guidance_opt.delta_t

        ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            if not self.ism:
                prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                target = noise
            else:
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t, starting_ind, inverse_text_embeddings, 
                                                                                guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=guidance_opt.xs_eta)
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t, inverse_text_embeddings, 
                                                                           guidance_opt.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True)        

                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]


        with torch.no_grad():
            latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
            if use_control_net:
                pred_depth_input = pred_depth_input[None, :, ...].repeat(1 + K, 1, 3, 1, 1).reshape(-1, 3, 512, 512).half()
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    latent_model_input,
                    tt,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=pred_depth_input,
                    return_dict=False,
                )
                unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings,
                                    down_block_additional_residuals=down_block_res_samples,
                                    mid_block_additional_residual=mid_block_res_sample).sample
            else:
                unet_output = self.unet(latent_model_input.to(self.precision_t), tt.to(self.precision_t), encoder_hidden_states=text_embeddings.to(self.precision_t)).sample

            unet_output = unet_output.reshape(2, -1, 4, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            delta_DSD = noise_pred_text - noise_pred_uncond
        
        pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD

        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5) 
    
        grad = w(self.alphas[t]) * (pred_noise - target)   

        grad_low, grad_high = self.decompose_latent_frequency(grad, kernel_size=5, opt_params=guidance_opt)
        # grad_low, grad_high = decompose_latent_frequency_torch(grad, kernel_size=5, sigma=1.0)
        grad_high = grad_high * 2.0
        
        current_t_val = t.item()
        if current_t_val > 400:
            sds_low_freq_scale = 0.5
        else:
            sds_low_freq_scale = 1.0
            
        grad_modified = grad_high + grad_low * sds_low_freq_scale

        grad_norm = torch.norm(grad_modified, dim=1, keepdim=True)
        max_grad_norm = 100.0 # 100.0
        grad_modified = grad_modified * torch.minimum(torch.ones_like(grad_norm), max_grad_norm / (grad_norm + 1e-8)) # * grad_scale
        grad_modified = torch.nan_to_num(grad_scale * grad_modified)

        loss = SpecifyGradient.apply(latents, grad_modified)

              
        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + 7.5* delta_DSD    
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,prev_t.item()))
            with torch.no_grad():
                d_raw = pred_depth.detach()
                a_raw = pred_alpha.detach()

                max_a = a_raw.max()
                thresh = max_a * 0.5
                mask = a_raw > thresh
                
                if mask.any():

                    obj_d = d_raw[mask]
                    v_min = torch.quantile(obj_d, 0.05)
                    v_max = torch.quantile(obj_d, 0.95)
                    enhanced_depth = (v_max - d_raw) / (v_max - v_min + 1e-7)
                    enhanced_depth = torch.clamp(enhanced_depth, 0.0, 1.0)

                    alpha_s = torch.clamp((a_raw - thresh) / (max_a - thresh + 1e-7), 0, 1)
                    enhanced_depth = enhanced_depth * alpha_s
                    enhanced_depth[~mask] = 0
                else:
                    enhanced_depth = torch.zeros_like(d_raw)

                grad_abs = torch.abs(grad.detach())
                norm_grad = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)
                        
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)    
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))


                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

                viz_images = torch.cat([pred_rgb, 
                                        enhanced_depth.repeat(1, 3, 1, 1),
                                        pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), 
                                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        latents_rgb, latents_sp_rgb, norm_grad,
                                        pred_x0_sp, pred_x0_pos],dim=0) 
                save_image(viz_images, save_path_iter)
                
            


        raw_sds_scaled = grad * grad_scale
        correction = grad_modified - raw_sds_scaled
        control_variate = raw_sds_scaled - grad_modified
        with torch.no_grad():
            raw_grad_norm = grad.norm().item()
            low_freq_norm = grad_low.norm().item()
            high_freq_norm = grad_high.norm().item()
            low_freq_ratio = low_freq_norm / (raw_grad_norm + 1e-8)
            
            if admm_external_force is not None:
                admm_control_std = admm_external_force.std().item()
            else:

                admm_control_std = 0.0

            monitor_dict = {
                "sds_norm": raw_grad_norm,
                "sds_final": grad_modified.norm().item(),
                "sds_low": low_freq_norm,
                "sds_high": high_freq_norm,
                "low_ratio": low_freq_ratio,
                "timestep": t.item(),
                "score_std": raw_sds_scaled.std().item(),
                "control_std": control_variate.std().item(), 
                "admm_control_std": admm_control_std,
                "score_mean": raw_sds_scaled.mean().item(),
                "control_mean": correction.mean().item()
            }

        return loss, monitor_dict

    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)

    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence
    
    def decompose_latent_frequency(self, latents, kernel_size=None, opt_params=None): 
        device = latents.device
        dtype = latents.dtype
        x_np = latents.detach().cpu().numpy()
        
        z_low_np = np.zeros_like(x_np)

        d = 5 
        sigmaColor = 0.05
        sigmaSpace = 5.0
        
        filter_type = getattr(opt_params, "pnp_filter_type", "bilateral")

        for b in range(x_np.shape[0]):
            for c in range(x_np.shape[1]):
                img = x_np[b, c]

                if filter_type == 'gaussian':
                    z_low_np[b, c] = cv2.GaussianBlur(img.astype(np.float32), (d, d), 0)
                elif filter_type == 'median':
                    z_low_np[b, c] = cv2.medianBlur(img.astype(np.float32), d)
                else: 
                    filtered = cv2.bilateralFilter(img.astype(np.float32), d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
                    z_low_np[b, c] = filtered

        z_low = torch.from_numpy(z_low_np).to(device=device, dtype=dtype)
        z_high = latents - z_low
        
        return z_low, z_high
    

    def apply_latent_mbir_prox(self, noisy_input, target_low_freq, alpha=0.5):
        z_low, z_high = self.decompose_latent_frequency(noisy_input)
        z_low_new = (1 - alpha) * z_low + alpha * target_low_freq

        z_new = z_low_new + z_high
        return z_new
    
    