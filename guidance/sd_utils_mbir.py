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


# ========== 在文件顶部，类定义之前添加 ==========

def depth_to_normal(depth_map):
    """
    从深度图计算法向图
    输入: depth_map [B, 1, H, W]
    输出: normal_map [B, 3, H, W] (归一化后的法向，范围 -1~1)
    """
    B, C, H, W = depth_map.shape
    
    # Sobel 算子计算梯度
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            device=depth_map.device, dtype=depth_map.dtype).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            device=depth_map.device, dtype=depth_map.dtype).view(1, 1, 3, 3)
    
    dx = F.conv2d(depth_map, kernel_x, padding=1)
    dy = F.conv2d(depth_map, kernel_y, padding=1)
    
    # 构造法向向量 (-dx, -dy, 1) 并归一化
    normal = torch.cat((-dx, -dy, torch.ones_like(depth_map)), dim=1)
    normal = F.normalize(normal, dim=1)
    
    return normal

def depth_to_normal_colored(depth_map, alpha_mask=None):
    """
    从深度图计算彩色法向图（标准 Normal Map 风格）
    
    输入: 
        depth_map: [B, 1, H, W] 或 [1, H, W]
        alpha_mask: [B, 1, H, W] 或 [1, H, W]，用于 mask 背景
    输出: 
        normal_rgb: [B, 3, H, W], 范围 0~1
    """
    # 处理维度
    if depth_map.dim() == 3:
        depth_map = depth_map.unsqueeze(0)
    if alpha_mask is not None and alpha_mask.dim() == 3:
        alpha_mask = alpha_mask.unsqueeze(0)
    
    B, C, H, W = depth_map.shape
    device, dtype = depth_map.device, depth_map.dtype
    
    # 1. 归一化 depth（关键！让梯度在合理范围内）
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    
    # 2. Sobel 算子计算梯度
    kernel_x = torch.tensor([[-1, 0, 1], 
                              [-2, 0, 2], 
                              [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], 
                              [ 0,  0,  0], 
                              [ 1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    
    dx = F.conv2d(F.pad(depth_norm, (1,1,1,1), mode='replicate'), kernel_x)
    dy = F.conv2d(F.pad(depth_norm, (1,1,1,1), mode='replicate'), kernel_y)
    
    # 3. 调整梯度强度（这个参数很关键！）
    # 值越大，法向变化越明显（颜色更丰富）
    gradient_scale = 2.0  # 可以尝试 1.0 ~ 5.0
    dx = dx * gradient_scale
    dy = dy * gradient_scale
    
    # 4. 构造法向向量 n = (-dx, -dy, 1)
    normal_x = -dx
    normal_y = -dy
    normal_z = torch.ones_like(dx)
    
    normal = torch.cat([normal_x, normal_y, normal_z], dim=1)  # [B, 3, H, W]
    
    # 5. 归一化
    normal = F.normalize(normal, dim=1, eps=1e-6)
    
    # 6. 映射到 RGB 颜色空间: [-1, 1] -> [0, 1]
    normal_rgb = (normal + 1.0) * 0.5
    
    # 7. 用 alpha mask 把背景变黑
    if alpha_mask is not None:
        # 二值化 alpha（阈值处理）
        alpha_binary = (alpha_mask > 0.5).float()
        normal_rgb = normal_rgb * alpha_binary
    
    return normal_rgb.clamp(0, 1)


def get_clean_normal(pred_normal):
    """
    使用双边滤波对法向图去噪
    输入: [B, 3, H, W], 范围 -1~1
    输出: [B, 3, H, W], 去噪后的法向图
    """
    normal_np = pred_normal.detach().permute(0, 2, 3, 1).cpu().numpy()
    # 映射到 0-255
    normal_np = ((normal_np + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    
    clean_np = np.zeros_like(normal_np)
    for b in range(normal_np.shape[0]):
        clean_np[b] = cv2.bilateralFilter(normal_np[b], d=5, sigmaColor=75, sigmaSpace=75)
    
    clean_normal = torch.from_numpy(clean_np).to(pred_normal.device).float() / 255.0
    clean_normal = (clean_normal * 2) - 1  # 映射回 -1~1
    return clean_normal.permute(0, 3, 1, 2)


def normal_to_rgb(normal):
    """
    将法向图转换为可视化的 RGB 图像
    输入: normal [B, 3, H, W], 范围 -1~1
    输出: rgb [B, 3, H, W], 范围 0~1
    """
    # 标准法向可视化：(normal + 1) / 2
    # X: 红色通道, Y: 绿色通道, Z: 蓝色通道
    rgb = (normal + 1) * 0.5
    return rgb.clamp(0, 1)

def normal_to_grayscale(normal):
    """
    将法向图转换为灰度可视化（类似深度图效果）
    
    输入: normal [B, 3, H, W], 范围 -1~1
    输出: gray [B, 1, H, W], 范围 0~1
    
    原理: 取 Z 分量，表示"朝向相机的程度"
          Z 越大（越朝向相机）越亮
    """
    # 取 Z 分量 (第三个通道)
    z_component = normal[:, 2:3, :, :]  # [B, 1, H, W]
    
    # 映射到 0~1 (原本 -1~1)
    gray = (z_component + 1) * 0.5
    
    return gray.clamp(0, 1)


def visualize_tensor_as_grayscale(tensor, normalize=True):
    """
    通用的灰度可视化函数
    
    输入: tensor [B, C, H, W]
    输出: gray [B, 1, H, W], 范围 0~1
    """
    # 取所有通道的均值
    if tensor.shape[1] > 1:
        gray = tensor.mean(dim=1, keepdim=True)
    else:
        gray = tensor
    
    if normalize:
        # 归一化到 0~1
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
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

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
        
        # =======================================================
        # [核心修改] Latent 梯度频域解耦 (Gradient Frequency Split)
        # =======================================================
        
        # 1. 分解梯度：算出 SDS 梯度的低频部分 (代表大的几何趋势)
        grad_low, grad_high = self.decompose_latent_frequency(grad, kernel_size=5, opt_params=guidance_opt)
        # grad_low, grad_high = decompose_latent_frequency_torch(grad, kernel_size=5, sigma=1.0)
        
        grad_high = grad_high * 2.0
        
        # 2. 抑制 SDS 的低频破坏力
        # 策略 A: 硬截断。如果当前的 timestep 比较大（前期），我们极度不信任 SDS 的几何
        # 策略 B: 软缩放。将低频梯度缩小 0.2 倍，给 ADMM 留出 80% 的话语权
        # (0.0, 0.2)
        current_t_val = t.item()
        # if current_t_val > 600:
        #     sds_low_freq_scale = 0.5  # 绝对禁言
        # else:
        #     sds_low_freq_scale = 1.0  # 允许微调
            
        if current_t_val > 500:
            sds_low_freq_scale = 0.1  # 高噪声时刻：强烈抑制（保护几何）
        elif current_t_val > 200:
            sds_low_freq_scale = 0.3  # 中等噪声：适度抑制
        else:
            sds_low_freq_scale = 0.6  # 低噪声时刻：放松限制（允许纹理精修）
        
        # sds_low_freq_scale = 0.2  # 这个参数决定了你跟 Benchmark 的差距有多大！
        #                         # 越小，几何越受 ADMM/几何约束控制，越平滑。
        
        grad_modified = grad_high + grad_low * sds_low_freq_scale
        
        # 3. 梯度归一化 (可选，防止梯度爆炸掩盖 ADMM)
        # 限制 SDS 梯度的最大模长，确保它不会比 ADMM 梯度大几个数量级
        max_grad_norm = 100 # （1.1，1.2为100，1.3为20，1.4为100）
        grad_norm = torch.norm(grad_modified, dim=1, keepdim=True)
        grad_modified = grad_modified * torch.minimum(torch.ones_like(grad_norm), max_grad_norm / (grad_norm + 1e-8)) # * grad_scale
        grad_modified = torch.nan_to_num(grad_scale * grad_modified)
        # =======================================================

        # 将修改后的梯度传给 SpecifyGradient
        # 注意：这里我们传 grad_modified，而不是原始的 grad
        loss = SpecifyGradient.apply(latents, grad_modified)
            
        # grad = torch.nan_to_num(grad_scale * grad)
        # loss = SpecifyGradient.apply(latents, grad)

        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD    
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,prev_t.item()))
            with torch.no_grad():
                d_raw = pred_depth.detach()
                a_raw = pred_alpha.detach()
                
                # 2. 深度图增强逻辑 (消除白边 + 层次拉伸)
                # 动态阈值：取当前视角 alpha 最大值的 50%
                max_a = a_raw.max()
                thresh = max_a * 0.5
                mask = a_raw > thresh
                
                if mask.any():
                    # 锁定物体核心区域的深度范围
                    obj_d = d_raw[mask]
                    v_min = torch.quantile(obj_d, 0.05) # 排除离群点
                    v_max = torch.quantile(obj_d, 0.95)
                    
                    # 线性拉伸并反转 (让近处白，远处黑)
                    # 如果你想要原样，就用 (d_raw - v_min) / (v_max - v_min)
                    enhanced_depth = (v_max - d_raw) / (v_max - v_min + 1e-7)
                    enhanced_depth = torch.clamp(enhanced_depth, 0.0, 1.0)
                    
                    # 消除白边：乘以 alpha 的权重消融
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
                
        # if iteration % guidance_opt.vis_interval == 0:
        #     noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD    
        #     lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
        #     save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,prev_t.item()))
            
        #     with torch.no_grad():
        #         # --- 1. 原有的 Latent 解码 ---
        #         pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)    
        #         pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
        #         pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
        #         pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))

        #         # --- Helper: 定义一个快速画热力图的函数 ---
        #         def get_heatmap(gradient_tensor):
        #             if gradient_tensor is None: 
        #                 return torch.zeros_like(pred_rgb) # 用于占位
        #             g_abs = torch.abs(gradient_tensor.detach())
        #             # 归一化到 0-1 方便观看
        #             g_norm = (g_abs / (g_abs.max() + 1e-8)).mean(dim=1, keepdim=True)
        #             return F.interpolate(g_norm, (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

        #         # --- 2. [关键新增] 梯度分量可视化 ---
        #         # A. 总梯度 (你原来的 norm_grad)
        #         viz_total_grad = get_heatmap(grad) 
                
        #         # B. 低频梯度 (SDS 想要怎么改几何) -> 论文图表素材!
        #         viz_low_freq = get_heatmap(grad_low)
                
        #         # C. 高频梯度 (SDS 想要怎么改纹理)
        #         viz_high_freq = get_heatmap(grad_high)
                
        #         # D. 被抑制的梯度 (SDS 想做但被我们禁止的操作) -> Janus Killer 的证据!
        #         # 只有当 scaling < 1.0 时才有意义
        #         viz_suppressed = get_heatmap((grad - grad_modified))

        #         # E. ADMM 外部力 (如果存在) -> 展示几何约束在哪里生效
        #         viz_admm_force = get_heatmap(admm_external_force) if admm_external_force is not None else torch.zeros_like(pred_rgb)

        #         # --- 3. Latent RGB 预览 ---
        #         latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
        #         latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

        #         # --- 4. 拼图 (Rows: 渲染 / 预测 / 梯度分析) ---
        #         # 建议分两行或三行，逻辑更清晰
                
        #         # Row 1: 现状 (RGB, Depth, Alpha, Latent)
        #         row1 = torch.cat([pred_rgb, pred_depth.repeat(1, 3, 1, 1), pred_alpha.repeat(1, 3, 1, 1), latents_rgb], dim=0)
                
        #         # Row 2: 预测 (X0_Pos: SDS想要的样子, X0_SP:原本的样子)
        #         row2 = torch.cat([pred_x0_pos, pred_x0_sp, latents_sp_rgb, torch.zeros_like(pred_rgb)], dim=0) # 补一个空位对齐
                
        #         # Row 3: [Paper Highlights] 梯度分析
        #         # 顺序: 低频(几何) | 高频(纹理) | 被抑制(Janus) | ADMM力(修正)
        #         row3 = torch.cat([viz_low_freq, viz_high_freq, viz_suppressed, viz_admm_force], dim=0)

        #         # 最终拼接
        #         viz_images = torch.cat([row1, row2, row3], dim=1) # 注意：这里为了方便看，改成 dim=1 横向拼可能会太长，建议先 cat row 再 cat dim=0
                
        #         # 修正拼接逻辑 (按你原来的 dim=0 竖着拼，但建议每行 4 张图)
        #         # 假设 B=1，我们手动拼成网格：
        #         # [ RGB  | Depth | Alpha | Latent ]
        #         # [ Pred | Low   | High  | ADMM   ]
                
        #         final_grid = torch.cat([
        #             torch.cat([pred_rgb, pred_depth.repeat(1,3,1,1), pred_alpha.repeat(1,3,1,1), latents_rgb], dim=3), # Row 1
        #             torch.cat([pred_x0_pos, viz_low_freq, viz_high_freq, viz_admm_force], dim=3)      # Row 2 (核心分析)
        #         ], dim=2)

        #         save_image(final_grid, save_path_iter)
                
                
        with torch.no_grad():
            # 计算各种模长 (Norm)
            raw_grad_norm = grad.norm().item()         # 原始 SDS 梯度的总力度
            low_freq_norm = grad_low.norm().item()     # 低频（几何）部分的力度
            high_freq_norm = grad_high.norm().item()   # 高频（纹理）部分的力度
            
            # 计算低频占比 (如果这个值很高，说明 SDS 还在疯狂修改几何)
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
                # 如果传进来了 ADMM 力，红线就是 ADMM 力度的 Std
                # 这代表：ADMM 这一步用了多大的力气来纠正几何
                admm_control_std = admm_external_force.std().item()
            else:
                # 前期没有 ADMM，红线归零
                admm_control_std = 0.0

            monitor_dict = {
                "sds_norm": raw_grad_norm,
                "sds_final": grad_modified.norm().item(),
                "sds_low": low_freq_norm,
                "sds_high": high_freq_norm,
                "low_ratio": low_freq_ratio,
                "timestep": t.item(),
                
                # [新增] 画图专用数据 (记录 Mean 值以观察方向)
                "score_std": raw_sds_scaled.std().item(),     # 蓝线: 原始 SDS 的混乱程度
                "control_std": control_variate.std().item(),  # 红线: 我们移除的混乱程度
                "admm_control_std": admm_control_std,
                
                "score_mean": raw_sds_scaled.mean().item(),     # 蓝线数据 (Score)
                "control_mean": correction.mean().item()     # 红线数据 (Control Variate)
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
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level

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
        
        # =======================================================
        # [核心修改] Latent 梯度频域解耦 (Gradient Frequency Split)
        # =======================================================
        
        # 1. 分解梯度：算出 SDS 梯度的低频部分 (代表大的几何趋势)
        grad_low, grad_high = self.decompose_latent_frequency(grad, kernel_size=5, opt_params=guidance_opt)
        # grad_low, grad_high = decompose_latent_frequency_torch(grad, kernel_size=5, sigma=1.0)
        grad_high = grad_high * 2.0
        
        # 2. 抑制 SDS 的低频破坏力
        # 策略 A: 硬截断。如果当前的 timestep 比较大（前期），我们极度不信任 SDS 的几何
        # 策略 B: 软缩放。将低频梯度缩小 0.2 倍，给 ADMM 留出 80% 的话语权
        
        current_t_val = t.item()
        if current_t_val > 400:
            sds_low_freq_scale = 0.5  # 绝对禁言
        else:
            sds_low_freq_scale = 1.0  # 允许微调
            
        # sds_low_freq_scale = 0.2  # 这个参数决定了你跟 Benchmark 的差距有多大！
        #                         # 越小，几何越受 ADMM/几何约束控制，越平滑。
        
        grad_modified = grad_high + grad_low * sds_low_freq_scale
        
        # 3. 梯度归一化 (可选，防止梯度爆炸掩盖 ADMM)
        # 限制 SDS 梯度的最大模长，确保它不会比 ADMM 梯度大几个数量级
        grad_norm = torch.norm(grad_modified, dim=1, keepdim=True)
        max_grad_norm = 100.0 # 100.0
        grad_modified = grad_modified * torch.minimum(torch.ones_like(grad_norm), max_grad_norm / (grad_norm + 1e-8)) # * grad_scale
        grad_modified = torch.nan_to_num(grad_scale * grad_modified)
        # =======================================================

        # 将修改后的梯度传给 SpecifyGradient
        # 注意：这里我们传 grad_modified，而不是原始的 grad
        loss = SpecifyGradient.apply(latents, grad_modified)


        # grad = torch.nan_to_num(grad_scale * grad)
        # loss = SpecifyGradient.apply(latents, grad)
              
        if iteration % guidance_opt.vis_interval == 0:
            noise_pred_post = noise_pred_uncond + 7.5* delta_DSD    
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,prev_t.item()))
            with torch.no_grad():
                d_raw = pred_depth.detach()
                a_raw = pred_alpha.detach()
                
                # 2. 深度图增强逻辑 (消除白边 + 层次拉伸)
                # 动态阈值：取当前视角 alpha 最大值的 50%
                max_a = a_raw.max()
                thresh = max_a * 0.5
                mask = a_raw > thresh
                
                if mask.any():
                    # 锁定物体核心区域的深度范围
                    obj_d = d_raw[mask]
                    v_min = torch.quantile(obj_d, 0.05) # 排除离群点
                    v_max = torch.quantile(obj_d, 0.95)
                    
                    # 线性拉伸并反转 (让近处白，远处黑)
                    # 如果你想要原样，就用 (d_raw - v_min) / (v_max - v_min)
                    enhanced_depth = (v_max - d_raw) / (v_max - v_min + 1e-7)
                    enhanced_depth = torch.clamp(enhanced_depth, 0.0, 1.0)
                    
                    # 消除白边：乘以 alpha 的权重消融
                    alpha_s = torch.clamp((a_raw - thresh) / (max_a - thresh + 1e-7), 0, 1)
                    enhanced_depth = enhanced_depth * alpha_s
                    enhanced_depth[~mask] = 0
                else:
                    enhanced_depth = torch.zeros_like(d_raw)

                # 3. 计算梯度可视化
                grad_abs = torch.abs(grad.detach())
                norm_grad = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)
                        
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)    
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)        
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))
                # pred_x0_uncond = pred_x0_sp[:1, ...]

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
                # 如果传进来了 ADMM 力，红线就是 ADMM 力度的 Std
                # 这代表：ADMM 这一步用了多大的力气来纠正几何
                admm_control_std = admm_external_force.std().item()
            else:
                # 前期没有 ADMM，红线归零
                admm_control_std = 0.0

            monitor_dict = {
                "sds_norm": raw_grad_norm,
                "sds_final": grad_modified.norm().item(),
                "sds_low": low_freq_norm,
                "sds_high": high_freq_norm,
                "low_ratio": low_freq_ratio,
                "timestep": t.item(),
                
                # [新增] 画图专用数据 (记录 Mean 值以观察方向)
                "score_std": raw_sds_scaled.std().item(),     # 蓝线: 原始 SDS 的混乱程度
                "control_std": control_variate.std().item(),  # 红线: 我们移除的混乱程度
                "admm_control_std": admm_control_std,
                "score_mean": raw_sds_scaled.mean().item(),     # 蓝线数据 (Score)
                "control_mean": correction.mean().item()     # 红线数据
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
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence
    
    # [新增] 频域分解工具
    # def decompose_latent_frequency(self, latents, kernel_size=3):
    #     """
    #     将 Latent 正交分解为: z = z_low (结构/流形) + z_high (纹理/细节)
    #     """
    #     # 使用 AvgPool 模拟低通滤波
    #     padding = kernel_size // 2
    #     z_low = F.avg_pool2d(latents, kernel_size=kernel_size, stride=1, padding=padding)
    #     z_high = latents - z_low
    #     return z_low, z_high
    
    def decompose_latent_frequency(self, latents, kernel_size=None, opt_params=None): 
        """
        [PnP-ADMM 升级版]
        使用双边滤波 (Bilateral Filter) 代替简单的 AvgPool。
        
        物理含义:
        z_low (Geometry) = 保持了边缘的平滑结构 (Edge-Preserving Smooth)
        z_high (Noise)   = 被剔除的随机噪声
        """
        device = latents.device
        dtype = latents.dtype
        
        # 1. 准备数据: Tensor (GPU) -> Numpy (CPU)
        # latents shape: [B, 4, H, W]
        x_np = latents.detach().cpu().numpy()
        
        z_low_np = np.zeros_like(x_np)
        
        # 双边滤波参数 (经验值，可微调)
        # d: 滤波直径 (对应 kernel_size)
        # sigmaColor: 颜色空间标准差 (控制对边缘的敏感度，越大越模糊，越小越保边)
        # sigmaSpace: 坐标空间标准差 (控制平滑范围)
        d = 5 
        sigmaColor = 0.05  # Latent 归一化后通常在 -2~2 之间，0.1 是个合理的边缘阈值
        sigmaSpace = 5.0
        
        filter_type = getattr(opt_params, "pnp_filter_type", "bilateral")
        
        # 2. 对每个 Sample 和 Channel 单独做滤波
        # OpenCV 不支持 4D 数组，必须循环处理
        for b in range(x_np.shape[0]):
            for c in range(x_np.shape[1]):
                img = x_np[b, c]
                
                # [核心] Bilateral Filter: 只磨平噪声，不磨平边缘
                # 注意: img 必须是 float32
                if filter_type == 'gaussian':
                    # 强力平滑，用于对比细节丢失
                    z_low_np[b, c] = cv2.GaussianBlur(img.astype(np.float32), (d, d), 0)
                elif filter_type == 'median':
                    # 中值滤波，用于对比对畸形突出的抑制
                    z_low_np[b, c] = cv2.medianBlur(img.astype(np.float32), d)
                else: 
                    filtered = cv2.bilateralFilter(img.astype(np.float32), d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
                    z_low_np[b, c] = filtered

        # 3. 转回 Tensor
        z_low = torch.from_numpy(z_low_np).to(device=device, dtype=dtype)
        
        # 4. 计算残差 (即 High Frequency / Noise)
        # 这一部分就是会被 ADMM 抑制掉的"多头"和"噪点"
        z_high = latents - z_low
        
        return z_low, z_high
    

    # [新增 2/2] MBIR 近邻算子 (即 ADMM 的 Z-step)
    def apply_latent_mbir_prox(self, noisy_input, target_low_freq, alpha=0.5):
        """
        求解 Z-step: min_z ||z - noisy_input||^2 + Constraint(z)
        noisy_input: 对应 ADMM 中的 (x + u)
        target: 对应几何流形的目标 (如 Batch 均值)
        alpha: 投影强度 (对应 rho/(rho+lambda))
        """
        z_low, z_high = self.decompose_latent_frequency(noisy_input)
        # z_low, z_high = decompose_latent_frequency_torch(noisy_input, kernel_size=5, sigma=1.0)
        
        # Proximal Update: 低频部分向目标收缩
        z_low_new = (1 - alpha) * z_low + alpha * target_low_freq
        
        # 重组: 保持高频部分不变 (因为我们只约束了低频流形)
        z_new = z_low_new + z_high
        return z_new
    
    