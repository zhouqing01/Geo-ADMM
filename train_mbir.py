#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import random
import imageio
import os
import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GenerateCamParams, GuidanceParams
import math
from torchvision.utils import save_image
import torchvision.transforms as T
from datetime import datetime
import torch.nn.functional as F
from eval_metrics import evaluate_finish
import numpy as np
from collections import deque
import cv2

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def calculate_multiscale_admm_loss(x, z, u, rho):
    loss = 0.0
    
    # Scale 1: 原始尺度 (64x64) - 负责细节对齐
    diff_scale1 = x - z + u
    loss += 1.0 * rho * (diff_scale1 ** 2).mean()
    
    # Scale 2: 下采样 2 倍 (32x32) - 负责中频结构
    x_32 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
    z_32 = F.interpolate(z, scale_factor=0.5, mode='bilinear', align_corners=False)
    u_32 = F.interpolate(u, scale_factor=0.5, mode='bilinear', align_corners=False)
    diff_scale2 = x_32 - z_32 + u_32
    loss += 0.5 * rho * 1.0 * (diff_scale2 ** 2).mean() 
    
    # Scale 3: 下采样 4 倍 (16x16) - 负责全局拓扑 (Janus Killer!)
    x_16 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
    z_16 = F.interpolate(z, scale_factor=0.25, mode='bilinear', align_corners=False)
    u_16 = F.interpolate(u, scale_factor=0.25, mode='bilinear', align_corners=False)
    diff_scale3 = x_16 - z_16 + u_16
    loss += 0.5 * rho * 4.0 * (diff_scale3 ** 2).mean()
    
    return loss 

def calculate_unified_admm_loss(x, z, u, rho):
    """
    修复版：直接接收 z (64x64)，内部自动构建金字塔。
    x: 当前生成的 Latent (64x64)
    z: 几何模板 (64x64) - 也就是你代码里的 clean_z
    u: 对偶变量 (64x64)
    """
    
    # 1. 内部构建 Z 的金字塔 (On-the-fly Pyramid Construction)
    # 我们认为 z (经过滤波的) 是最可靠的几何参考
    z_64 = z
    z_16 = F.interpolate(z, scale_factor=0.25, mode='bilinear', align_corners=False)
    
    # 2. 提取“骨架” (Base Structure) - 消除多头
    # 将 16x16 的低频信息强制拉回 64x64。这个版本只有大轮廓，绝对没有细碎的多头。
    z_base_upsampled = F.interpolate(z_16, size=x.shape[2:], mode='bilinear', align_corners=False)
    
    # 3. 提取“细节” (High-Freq Detail) - 保留锐度
    # 我们使用 64x64 的 z 中的高频纹理
    # 简单的“原图 - 模糊图”提取高频
    z_blurred = F.avg_pool2d(z_64, kernel_size=5, stride=1, padding=2)
    z_detail = z_64 - z_blurred
    
    # 4. 合成“完美目标” (Unified Target)
    # 核心公式：Target = 低频骨架 (来自 Scale 3) + 高频皮肤 (来自 Scale 1)
    # alpha 越大，越强行纠正形状 (防多头)；越小，越保留原始细节。
    alpha = 0.8 
    
    # 注意：这里的 x_blurred 其实不需要了，我们完全信任 z 的结构
    # 但为了平滑过渡，可以用 z_base 和 z_detail 合成
    z_unified = alpha * z_base_upsampled + (1 - alpha) * z_blurred + z_detail
    
    # 5. 计算 Loss (只在 64x64 这一层算，避免打架)
    # x 必须去逼近这个“合成出来的完美 z”
    loss = 0.5 * rho * ((x - z_unified + u) ** 2).mean()
    
    return loss

def calculate_coarse_admm_loss(x, z, u, rho, iteration, opt):
    """
    只在粗尺度上计算 ADMM loss，避免空间错位
    """
    progress = (iteration - opt.geo_start) / max(opt.iterations - opt.geo_start, 1)
    
    # 渐进式尺度：前期更粗，后期稍细
    if progress < 0.4:
        target_size = (8, 8)    # 极粗：只管拓扑
        weight = 4.0
    elif progress < 0.8:
        target_size = (16, 16)  # 中等：管形状
        weight = 2.0
    else:
        target_size = (32, 32)  # 较细：允许结构细节
        weight = 1.0
    
    x_scaled = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    z_scaled = F.interpolate(z, size=target_size, mode='bilinear', align_corners=False)
    u_scaled = F.interpolate(u, size=target_size, mode='bilinear', align_corners=False)
    
    diff = x_scaled - z_scaled + u_scaled
    loss = weight * 0.5 * rho * (diff ** 2).mean()
    
    return loss


def get_clean_normal_pnp(pred_normal):
    """输入 [B, 3, H, W], 输出去噪后的 [B, 3, H, W]"""
    normal_np = pred_normal.detach().permute(0, 2, 3, 1).cpu().numpy()
    # 映射到 0-255
    normal_np = ((normal_np + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    
    clean_np = np.zeros_like(normal_np)
    for b in range(normal_np.shape[0]):
        # 双边滤波: 保边去噪
        clean_np[b] = cv2.bilateralFilter(normal_np[b], d=5, sigmaColor=75, sigmaSpace=75)
        
    clean_normal = torch.from_numpy(clean_np).to(pred_normal.device).float() / 255.0
    # 映射回 -1~1 (如果你的 Normal 是这个范围) 或 0~1
    clean_normal = (clean_normal * 2) - 1 
    return clean_normal.permute(0, 3, 1, 2)

def depth_to_normal(depth_map):
    """
    输入: depth_map [B, 1, H, W]
    输出: normal_map [B, 3, H, W] (归一化后的法向)
    """
    B, C, H, W = depth_map.shape
    
    # 1. 计算梯度 (使用简单的差分算子)
    # dy: 垂直方向梯度 (下 - 上)
    # dx: 水平方向梯度 (右 - 左)
    # 为了保持 tensor 尺寸一致，我们用 padding
    
    # 简单的 Sobel 或者 差分卷积核
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=depth_map.device, dtype=depth_map.dtype).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=depth_map.device, dtype=depth_map.dtype).view(1, 1, 3, 3)
    
    # Padding 1 保持尺寸不变
    dx = F.conv2d(depth_map, kernel_x, padding=1)
    dy = F.conv2d(depth_map, kernel_y, padding=1)
    
    # 2. 构造法向向量 (-dx, -dy, 1)
    # 注意: 这里假设是右手坐标系，相机看向 -Z 或 +Z 需根据实际情况调整符号
    # 但对于 ADMM 平滑约束来说，符号反了不影响平滑性，只要一致就行
    normal = torch.cat((-dx, -dy, torch.ones_like(depth_map)), dim=1)
    
    # 3. 归一化 (Normalize)
    normal = F.normalize(normal, dim=1)
    
    return normal 
    
# [新增类] ADMM 状态管理器
class ADMMStateManager:
    def __init__(self):
        # 显存优化: 使用 CPU 存储历史状态，需要时才搬到 GPU
        self.cache = {} 
    
    def get_states(self, cam_uids, shape, device, dtype):
        """获取当前 Batch 对应的 u 和 z"""
        u_batch = []
        z_batch = []
        
        for uid in cam_uids:
            if uid not in self.cache:
                # 初始化: u=0, z=0
                self.cache[uid] = {
                    'u': torch.zeros(shape, dtype=dtype, device='cpu'), # 存 CPU
                    'z': torch.zeros(shape, dtype=dtype, device='cpu')
                }
            # 搬运到 GPU 参与计算
            u_batch.append(self.cache[uid]['u'].to(device))
            z_batch.append(self.cache[uid]['z'].to(device))
            
        return torch.stack(u_batch), torch.stack(z_batch)

    def update_states(self, cam_uids, u_new, z_new):
        """更新并持久化"""
        u_new = u_new.detach().cpu() # 存回 CPU 省显存
        z_new = z_new.detach().cpu()
        
        for i, uid in enumerate(cam_uids):
            self.cache[uid]['u'] = u_new[i]
            self.cache[uid]['z'] = z_new[i]
            
class ADMMStateManager_LowFreq:
    """
    [论文核心] 低频域 ADMM 状态管理器
    只存储和操作低频分量，避免空间错位
    """
    def __init__(self, momentum=0.9):
        self.cache = {}
        self.momentum = momentum  # 时序平滑
    
    def get_states(self, cam_uids, shape, device, dtype):
        """获取低频空间的 u 和 z"""
        u_batch = []
        z_batch = []
        
        for uid in cam_uids:
            if uid not in self.cache:
                self.cache[uid] = {
                    'u_low': torch.zeros(shape, dtype=dtype, device='cpu'),
                    'z_low': None  # 延迟初始化
                }
            
            u_batch.append(self.cache[uid]['u_low'].to(device))
            
            if self.cache[uid]['z_low'] is not None:
                z_batch.append(self.cache[uid]['z_low'].to(device))
            else:
                z_batch.append(None)
        
        u_stacked = torch.stack(u_batch)
        return u_stacked, z_batch  # z_batch 是 list，可能含 None
    
    def update_states(self, cam_uids, u_low_new, z_low_new):
        """更新低频空间的状态"""
        u_low_new = u_low_new.detach().cpu()
        z_low_new = z_low_new.detach().cpu()
        
        for i, uid in enumerate(cam_uids):
            # 更新 u（直接覆盖）
            self.cache[uid]['u_low'] = u_low_new[i]
            
            # 更新 z（带动量平滑）
            if self.cache[uid]['z_low'] is not None:
                self.cache[uid]['z_low'] = (
                    self.momentum * self.cache[uid]['z_low'] + 
                    (1 - self.momentum) * z_low_new[i]
                )
            else:
                self.cache[uid]['z_low'] = z_low_new[i]
                
class StatisticalADMMManager:
    """统计量空间的 ADMM 状态管理器"""
    
    def __init__(self, stat_dim, momentum=0.9):
        self.cache = {}
        self.momentum = momentum
        self.stat_dim = stat_dim
    
    def get_states(self, cam_uids, device, dtype):
        u_batch = []
        z_batch = []
        
        for uid in cam_uids:
            if uid not in self.cache:
                self.cache[uid] = {
                    'u': torch.zeros(self.stat_dim, dtype=dtype, device='cpu'),
                    'z': None
                }
            
            u_batch.append(self.cache[uid]['u'].to(device))
            z_batch.append(self.cache[uid]['z'].to(device) if self.cache[uid]['z'] is not None else None)
        
        u_stacked = torch.stack(u_batch)  # [B, stat_dim]
        return u_stacked, z_batch
    
    def update_states(self, cam_uids, u_new, z_new):
        u_new = u_new.detach().cpu()
        z_new = z_new.detach().cpu()
        
        for i, uid in enumerate(cam_uids):
            self.cache[uid]['u'] = u_new[i]
            
            if self.cache[uid]['z'] is not None:
                self.cache[uid]['z'] = self.momentum * self.cache[uid]['z'] + (1 - self.momentum) * z_new[i]
            else:
                self.cache[uid]['z'] = z_new[i]


def extract_statistics(x_low, coarse_size=4):
    """
    提取统计量向量 S(x)
    输入: x_low [B, C, H, W]
    输出: stats [B, stat_dim]
    """
    B, C, H, W = x_low.shape
    
    # 1. 能量 [B, 1]
    energy = (x_low ** 2).mean(dim=[1, 2, 3], keepdim=False).unsqueeze(1)
    
    # 2. 通道均值 [B, C]
    ch_mean = x_low.mean(dim=[2, 3])
    
    # 3. 通道方差 [B, C]
    ch_var = x_low.var(dim=[2, 3])
    
    # 4. 极粗尺度特征 [B, C * coarse_size * coarse_size]
    x_coarse = F.interpolate(x_low, size=(coarse_size, coarse_size), mode='bilinear', align_corners=False)
    coarse_feat = x_coarse.view(B, -1)
    
    # 拼接成统计量向量
    stats = torch.cat([energy, ch_mean, ch_var, coarse_feat], dim=1)  # [B, 1 + C + C + C*16]
    
    return stats


def statistical_pnp_denoiser(stats, batch_stats, alpha=0.3):
    """
    统计量空间的 PnP 去噪器
    将当前统计量向 batch 一致性目标靠拢
    """
    # Batch 均值作为一致性目标
    global_target = batch_stats.mean(dim=0, keepdim=True).expand_as(stats)
    
    # 混合
    z_new = (1 - alpha) * stats + alpha * global_target
    
    return z_new


def adjust_text_embeddings(embeddings, azimuth, guidance_opt):
    #TODO: add prenerg functions
    text_z_list = []
    weights_list = []
    K = 0
    #for b in range(azimuth):
    text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth, guidance_opt)
    K = max(K, weights_.shape[0])
    text_z_list.append(text_z_)
    weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0) # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0) # [B * K]
    return text_embeddings, weights

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def prepare_embeddings(guidance_opt, guidance):
    embeddings = {}
    # text embeddings (stable-diffusion) and (IF)
    embeddings['default'] = guidance.get_text_embeds([guidance_opt.text])
    embeddings['uncond'] = guidance.get_text_embeds([guidance_opt.negative])

    for d in ['front', 'side', 'back']:
        embeddings[d] = guidance.get_text_embeds([f"{guidance_opt.text}, {d} view"])
    embeddings['inverse_text'] = guidance.get_text_embeds(guidance_opt.inverse_text)
    return embeddings

def guidance_setup(guidance_opt):
    if guidance_opt.guidance=="SD":
        from guidance.sd_utils_mbir import StableDiffusion
        guidance = StableDiffusion(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O, 
                                   guidance_opt.t_range, guidance_opt.max_t_range, 
                                   num_train_timesteps=guidance_opt.num_train_timesteps, 
                                   ddim_inv=guidance_opt.ddim_inv,
                                   textual_inversion_path = guidance_opt.textual_inversion_path,
                                   LoRA_path = guidance_opt.LoRA_path,
                                   guidance_opt=guidance_opt)
    else:
        raise ValueError(f'{guidance_opt.guidance} not supported.')
    if guidance is not None:
        for p in guidance.parameters():
            p.requires_grad = False
    embeddings = prepare_embeddings(guidance_opt, guidance)
    return guidance, embeddings


def training(dataset, opt, pipe, gcams, guidance_opt, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_video):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gcams, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    #
    save_folder = os.path.join(dataset._model_path,"train_process/")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs
        print('train_process is in :', save_folder)
    #controlnet
    use_control_net = False
    #set up pretrain diffusion models and text_embedings 
    guidance, embeddings = guidance_setup(guidance_opt)   
    viewpoint_stack = None
    viewpoint_stack_around = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if opt.save_process:
        save_folder_proc = os.path.join(scene.args._model_path,"process_videos/")
        if not os.path.exists(save_folder_proc):
            os.makedirs(save_folder_proc)  # makedirs
        process_view_points = scene.getCircleVideoCameras(batch_size=opt.pro_frames_num,render45=opt.pro_render_45).copy()    
        save_process_iter = opt.iterations // len(process_view_points)
        pro_img_frames = []
    
    C = 4  # latent channels
    coarse_size = 4
    stat_dim = 1 + C + C + C * coarse_size * coarse_size  # 能量 + 均值 + 方差 + 粗特征
    stat_admm_manager = StatisticalADMMManager(stat_dim=stat_dim, momentum=0.85)
    
    admm_manager = ADMMStateManager()
    # admm_manager = ADMMStateManager_new()
    # admm_manager = ADMMStateManager_LowFreq(momentum=0.85)
    rho = 5.0  # ADMM 惩罚参数，建议从 1.0 开始调
    loss_history = []

    for iteration in range(first_iter, opt.iterations + 1):        
        #TODO: DEBUG NETWORK_GUI
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, guidance_opt.text)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.update_feature_learning_rate(iteration)
        gaussians.update_rotation_learning_rate(iteration)
        gaussians.update_scaling_learning_rate(iteration)
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # progressively relaxing view range    
        if not opt.use_progressive:                
            if iteration >= opt.progressive_view_iter and iteration % opt.scale_up_cameras_iter == 0:
                scene.pose_args.fovy_range[0] = max(scene.pose_args.max_fovy_range[0], scene.pose_args.fovy_range[0] * opt.fovy_scale_up_factor[0])
                scene.pose_args.fovy_range[1] = min(scene.pose_args.max_fovy_range[1], scene.pose_args.fovy_range[1] * opt.fovy_scale_up_factor[1])

                scene.pose_args.radius_range[1] = max(scene.pose_args.max_radius_range[1], scene.pose_args.radius_range[1] * opt.scale_up_factor)
                scene.pose_args.radius_range[0] = max(scene.pose_args.max_radius_range[0], scene.pose_args.radius_range[0] * opt.scale_up_factor)

                scene.pose_args.theta_range[1] = min(scene.pose_args.max_theta_range[1], scene.pose_args.theta_range[1] * opt.phi_scale_up_factor)
                scene.pose_args.theta_range[0] = max(scene.pose_args.max_theta_range[0], scene.pose_args.theta_range[0] * 1/opt.phi_scale_up_factor)

                # opt.reset_resnet_iter = max(500, opt.reset_resnet_iter // 1.25)
                scene.pose_args.phi_range[0] = max(scene.pose_args.max_phi_range[0] , scene.pose_args.phi_range[0] * opt.phi_scale_up_factor)
                scene.pose_args.phi_range[1] = min(scene.pose_args.max_phi_range[1], scene.pose_args.phi_range[1] * opt.phi_scale_up_factor)
                
                print('scale up theta_range to:', scene.pose_args.theta_range)
                print('scale up radius_range to:', scene.pose_args.radius_range)
                print('scale up phi_range to:', scene.pose_args.phi_range)
                print('scale up fovy_range to:', scene.pose_args.fovy_range)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getRandTrainCameras().copy()         
        
        C_batch_size = guidance_opt.C_batch_size
        viewpoint_cams = []
        images = []
        text_z_ = []
        weights_ = []
        depths = []
        alphas = []
        scales = []
        pred_normals = []
        

        text_z_inverse =torch.cat([embeddings['uncond'],embeddings['inverse_text']], dim=0)

        for i in range(C_batch_size):
            try:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))            
            except:
                viewpoint_stack = scene.getRandTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                
            #pred text_z
            azimuth = viewpoint_cam.delta_azimuth
            text_z = [embeddings['uncond']]


            if guidance_opt.perpneg:
                text_z_comp, weights = adjust_text_embeddings(embeddings, azimuth, guidance_opt)
                text_z.append(text_z_comp)
                weights_.append(weights)

            else:                
                if azimuth >= -90 and azimuth < 90:
                    if azimuth >= 0:
                        r = 1 - azimuth / 90
                    else:
                        r = 1 + azimuth / 90
                    start_z = embeddings['front']
                    end_z = embeddings['side']
                else:
                    if azimuth >= 0:
                        r = 1 - (azimuth - 90) / 90
                    else:
                        r = 1 + (azimuth + 90) / 90
                    start_z = embeddings['side']
                    end_z = embeddings['back']
                text_z.append(r * start_z + (1 - r) * end_z)

            text_z = torch.cat(text_z, dim=0)
            text_z_.append(text_z)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, 
                                sh_deg_aug_ratio = dataset.sh_deg_aug_ratio, 
                                bg_aug_ratio = dataset.bg_aug_ratio, 
                                shs_aug_ratio = dataset.shs_aug_ratio, 
                                scale_aug_ratio = dataset.scale_aug_ratio)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]
        
            
            if depth.dim() == 3:
                depth_input = depth.unsqueeze(0)
            else:
                depth_input = depth
                
            # 实时计算伪法向
            pred_normal = depth_to_normal(depth_input)

            scales.append(render_pkg["scales"])
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)
            viewpoint_cams.append(viewpoint_cam)
            pred_normals.append(pred_normal.squeeze(0))

        images = torch.stack(images, dim=0)
        depths = torch.stack(depths, dim=0)
        alphas = torch.stack(alphas, dim=0)
        
        # =========================================================
        # [START] Rigorous ADMM Logic
        # =========================================================
        
        # 获取当前 Batch 的相机 ID
        batch_uids = [cam.uid for cam in viewpoint_cams]
        # batch_uids = ["global"] * len(viewpoint_cams)
        
        # 1. 显式编码得到 x (Current Latent)
        # 我们需要它的梯度，所以不能 no_grad
        x_latents, _ = guidance.encode_imgs(images.to(guidance.precision_t))
        
        if len(pred_normals) > 0:
            batch_pred_normals = torch.stack(pred_normals, dim=0).to(x_latents.device)
        else:
            batch_pred_normals = None
        
        loss_admm_coupling = 0.0
        loss_normal_admm = 0.0
        override_latents_for_sds = None
        
        # 仅在几何优化阶段开启 ADMM (后期)
        if opt.use_latent_admm and iteration > opt.geo_start: 
            with torch.no_grad():
                # --- Step 0: 获取历史状态 u^k, z^k ---
                u_old, z_old = admm_manager.get_states(
                    batch_uids, x_latents.shape[1:], x_latents.device, x_latents.dtype
                )
                
                # --- Step 1: Z-Minimization (Proximal Step) ---
                # 输入 v = x + u
                v_input = x_latents.detach() + u_old
                
                # # 构造流形目标 (Target): Batch 内低频一致性
                # # 这里我们调用 decompose 来计算 target
                # v_low, _ = guidance.decompose_latent_frequency(v_input)
                # target_low = v_low.mean(dim=0, keepdim=True).repeat(v_low.shape[0], 1, 1, 1)
                
                
                # # 调用 sd_utils 里的 prox 算子计算 z^{k+1}
                # # 这里复用了 apply_latent_mbir_prox
                # # z_new = guidance.apply_latent_mbir_prox(
                # #     noisy_input=v_input,
                # #     target_low_freq=target_low,
                # #     alpha=0.5 # 投影强度
                # # )
                
                # z_candidate = guidance.apply_latent_mbir_prox(
                #     noisy_input=v_input,
                #     target_low_freq=target_low,
                #     alpha=0.5 
                # )
                
                
                v_low, v_high = guidance.decompose_latent_frequency(v_input)
                
                global_target_low = v_low.mean(dim=0, keepdim=True).expand_as(v_low)
                
                is_initialized = (z_old.abs().sum() > 1e-6)
                
                # 动量设置：随着训练进行，越来越信任历史几何
                progress = min((iteration - opt.geo_start) / 2000.0, 1.0)
                momentum = 0.3 + 0.6 * progress # 0.3 -> 0.9
                
                if is_initialized:
                    # 1. 对低频 (几何) 施加动量约束 -> 解决 Janus 多头问题
                    # 我们希望几何结构稳定，不要乱变
                    z_old_low, _ = guidance.decompose_latent_frequency(z_old)
                    target_low = momentum * z_old_low + (1 - momentum) * global_target_low
                    # z_new_low = momentum * z_old_low + (1 - momentum) * v_low
                else:
                    target_low = global_target_low
                    # z_new_low = v_low

                # 3. 重新组合：z = 稳定的几何 + 清晰的纹理
                
                alpha = getattr(opt, "admm_alpha", 0.3)
                z_low_new = (1 - alpha) * v_low + alpha * target_low
                
                z_new = z_low_new + v_high
                
                # z_candidate, _ = guidance.decompose_latent_frequency(v_input)
                
                # is_initialized = (z_old.abs().sum() > 1e-6)
                
                # if is_initialized:
                #     # [策略 A] 动量法 (Momentum) - 推荐，最稳
                #     # 相当于软性的滑动窗口。0.8 代表 80% 相信历史，20% 相信当前
                #     momentum = 0.3 # 0.8
                #     z_new = momentum * z_old + (1 - momentum) * z_candidate
                    
                #     # [策略 B] 如果你坚持完全依赖 Manager 的 deque 均值
                #     # z_new = z_candidate 
                #     # 这种写法有滞后性：当前步 update u 用的是 raw，下一步才用到 mean。
                #     # 建议用策略 A，它能让当前的 u_update 立刻享受到平滑的好处。
                # else:
                #     z_new = z_candidate

                # --- Step 2: U-Update (Dual Ascent) ---
                # u^{k+1} = u^k + x^k - z^{k+1}
                u_new = u_old + x_latents.detach() - z_new
                
                # 截断防止爆炸 (Safety Clip)
                u_new = torch.clamp(u_new, -2.0, 2.0)
                
                # 保存状态
                admm_manager.update_states(batch_uids, u_new, z_new)
            
            # --- Step 3 Prepare: 计算 X-Step 的耦合项 ---
            # X-Step 需要最小化: L_SDS + (rho/2)||x - (z-u)||^2
            # 也就是把 x 拉向 (z - u)
            target_x = (z_new - u_old).detach()
            
            # # 计算 ADMM 物理约束 Loss
            # loss_admm_coupling = 0.5 * rho * F.mse_loss(x_latents, target_x)
            
            # loss_admm_coupling = calculate_multiscale_admm_loss(
            #     x=x_latents, 
            #     z=z_new.detach(), # 靶子必须 detach
            #     u=u_old.detach(), # 对偶变量也 detach
            #     rho=rho
            # )
        
            loss_admm_coupling = calculate_unified_admm_loss(
                x=x_latents, 
                z=z_new.detach(), # 靶子必须 detach
                u=u_old.detach(), # 对偶变量也 detach
                rho=rho
            )
            
            if batch_pred_normals is not None:
                # PnP 去噪: 获取"干净的法向" (调用 get_clean_normal_pnp)
                # 这一步会磨平法向图上的噪点
                target_normal = get_clean_normal_pnp(batch_pred_normals)
                
                # 计算 Loss (Huber Loss 更稳健)
                # 权重建议 0.5 - 1.0，这是直接作用于物理表面的强约束
                loss_normal_admm = 10000 * F.huber_loss(batch_pred_normals, target_normal, delta=0.1)
            
            admm_offset = x_latents.detach() - target_x
            admm_force_tensor = rho * admm_offset
            
            # 对于 SDS，我们传入原始的 x_latents
            # 因为 ADMM Loss 已经负责了"拉扯"，SDS 只需要负责"像真的"
            # 这是一个关键点：严谨 ADMM 中，SDS 和 Coupling 是加和关系
            # override_latents_for_sds = target_x 
            override_latents_for_sds = x_latents 
            
        else:
            # 前期：没有 ADMM，只有 SDS，需要传入 Latent
            override_latents_for_sds = x_latents
            
            admm_force_tensor = torch.zeros_like(x_latents)

        # =========================================================
        # [END] ADMM Logic
        # =========================================================
        
        
        if iteration > opt.geo_start:
            with torch.no_grad():
                u_old, z_old = admm_manager.get_states(
                    batch_uids, x_latents.shape[1:], x_latents.device, x_latents.dtype
                )
                
                v_input = x_latents.detach() + u_old
                
                # ========== 核心修正：在粗尺度上计算一致性目标 ==========
                
                # 1. 先下采样到安全尺度 (8x8)，消除空间错位
                v_coarse = F.interpolate(v_input, size=(8, 8), mode='bilinear', align_corners=False)
                
                # 2. 在粗尺度上做 batch 均值（这是安全的！）
                global_target_coarse = v_coarse.mean(dim=0, keepdim=True).expand_as(v_coarse)
                
                # 3. 上采样回 64x64（模糊的全局拓扑）
                global_target_64 = F.interpolate(global_target_coarse, size=x_latents.shape[2:], 
                                                mode='bilinear', align_corners=False)
                
                # 4. 分解当前 latent 的高低频
                v_low, v_high = guidance.decompose_latent_frequency(v_input)
                
                # 5. 只约束低频部分向"粗尺度一致性目标"靠拢
                # 高频(纹理)完全保留，不参与跨视角融合
                progress = min((iteration - opt.geo_start) / 2000.0, 1.0)
                momentum = 0.3 + 0.6 * progress
                
                is_initialized = (z_old.abs().sum() > 1e-6)
                
                if is_initialized:
                    z_old_low, _ = guidance.decompose_latent_frequency(z_old)
                    # 低频目标 = 历史稳定性 + 粗尺度全局一致性
                    target_low = momentum * z_old_low + (1 - momentum) * global_target_64
                else:
                    target_low = global_target_64
                
                alpha = getattr(opt, "admm_alpha", 0.3)
                z_low_new = (1 - alpha) * v_low + alpha * target_low
                
                # 重组：稳定的低频 + 原始的高频
                z_new = z_low_new + v_high
                
                # U-step
                u_new = u_old + x_latents.detach() - z_new
                u_new = torch.clamp(u_new, -2.0, 2.0)
                
                admm_manager.update_states(batch_uids, u_new, z_new)
            
            # Loss 计算也改用粗尺度
            loss_admm_coupling = calculate_coarse_admm_loss(
                x_latents, z_new.detach(), u_old.detach(), rho, iteration, opt
            )
            
            if opt.use_normal_red and batch_pred_normals is not None:
                # PnP 去噪: 获取"干净的法向" (调用 get_clean_normal_pnp)
                # 这一步会磨平法向图上的噪点
                target_normal = get_clean_normal_pnp(batch_pred_normals)
                
                # 计算 Loss (Huber Loss 更稳健)
                # 权重建议 0.5 - 1.0，这是直接作用于物理表面的强约束
                loss_normal_admm = 10000 * F.huber_loss(batch_pred_normals, target_normal, delta=0.1)
            
            admm_offset = x_latents.detach() - target_x
            admm_force_tensor = rho * admm_offset
            
            # 对于 SDS，我们传入原始的 x_latents
            # 因为 ADMM Loss 已经负责了"拉扯"，SDS 只需要负责"像真的"
            # 这是一个关键点：严谨 ADMM 中，SDS 和 Coupling 是加和关系
            # override_latents_for_sds = target_x 
            override_latents_for_sds = x_latents 
            
        else:
            # 前期：没有 ADMM，只有 SDS，需要传入 Latent
            override_latents_for_sds = x_latents
            
            admm_force_tensor = torch.zeros_like(x_latents)


        # Loss
        logs = {}
        warm_up_rate = 1. - min(iteration/opt.warmup_iter,1.)
        guidance_scale = guidance_opt.guidance_scale
        _aslatent = False
        if iteration < opt.geo_iter or random.random()< opt.as_latent_ratio:
            _aslatent=True
        if iteration > opt.use_control_net_iter and (random.random() < guidance_opt.controlnet_ratio):
                use_control_net = True
        if guidance_opt.perpneg:
            loss, logs = guidance.train_step_perpneg(torch.stack(text_z_, dim=1), images, 
                                                pred_depth=depths, pred_alpha=alphas,
                                                grad_scale=guidance_opt.lambda_guidance,
                                                use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                                weights = torch.stack(weights_, dim=1), resolution=(gcams.image_h, gcams.image_w),
                                                guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse, override_latents=override_latents_for_sds, admm_external_force=admm_force_tensor)
        else:
            loss, logs = guidance.train_step(torch.stack(text_z_, dim=1), images, 
                                    pred_depth=depths, pred_alpha=alphas,
                                    grad_scale=guidance_opt.lambda_guidance,
                                    use_control_net = use_control_net ,save_folder = save_folder,  iteration = iteration, warm_up_rate=warm_up_rate, 
                                    resolution=(gcams.image_h, gcams.image_w),
                                    guidance_opt=guidance_opt,as_latent=_aslatent, embedding_inverse = text_z_inverse, override_latents=override_latents_for_sds, admm_external_force=admm_force_tensor)
            #raise ValueError(f'original version not supported.')
        scales = torch.stack(scales, dim=0)

        loss_scale = torch.mean(scales,dim=-1).mean()
        loss_tv = tv_loss(images) + tv_loss(depths) 
        # loss_bin = torch.mean(torch.min(alphas - 0.0001, 1 - alphas))

        loss = loss + opt.lambda_tv * loss_tv + opt.lambda_scale * loss_scale + loss_admm_coupling + loss_normal_admm #opt.lambda_tv * loss_tv + opt.lambda_bin * loss_bin + opt.lambda_scale * loss_scale +
        loss.backward()
        iter_end.record()
        
        history_item = {
            "iter": iteration,
            "t": logs['timestep'],
            "score_mean": logs['score_mean'],       # 蓝线数据
            "control_mean": logs['control_mean'], # 红线数据
            "score_std": logs['score_std'],       
            "control_std": logs['control_std'],
            "admm_control_std": logs['admm_control_std'],
        }
        loss_history.append(history_item)
        
        if iteration % 500 == 0:
            save_path = os.path.join(scene.args._model_path, "gradient_history.npy")
            np.save(save_path, loss_history)
            print(f"[INFO] Gradient history saved to {save_path}")
        
        if iteration % 100 == 0:
            # 获取 ADMM Loss 的数值（如果是 tensor）
            admm_val = loss_admm_coupling.item() if isinstance(loss_admm_coupling, torch.Tensor) else loss_admm_coupling
            
            normal_val = loss_normal_admm.item() if isinstance(loss_normal_admm, torch.Tensor) else loss_normal_admm
            
            # 计算 "对抗比率"：SDS 的梯度力度 vs ADMM 的拉力
            # 如果 SDS 力度是 10.0，ADMM 是 0.001，那 ADMM 毫无意义
            # 这里的 admm_val 已经是 Loss，我们需要它的梯度量级估算，通常 Loss * 10 约等于梯度回传量级(粗略)
            
            # print(f"\n[DIAGNOSIS Step {iteration}] -----------------------------")
            # print(f"  > Time Step (t)    : {logs['timestep']:.4f}")
            # print(f"  > SDS Grad Norm    : {logs['sds_norm']:.4f} (总推力)")
            # print(f"  > SDS Grad final    : {logs['sds_final']:.4f} (最终传入的总推力)")
            # print(f"  > SDS Low Freq     : {logs['sds_low']:.4f} (几何破坏力)")
            # print(f"  > SDS High Freq    : {logs['sds_high']:.4f} (纹理精细力)")
            # print(f"  > Low Freq Ratio   : {logs['low_ratio']:.2%} (越低越好，说明几何被保护了)")
            # print(f"  ---------------------------------------------------")
            # print(f"  > ADMM Loss        : {admm_val:.6f} (几何约束力)")
            # print(f"  > ADMM Normal      : {normal_val:.6f} (法向/表面硬约束) <--- [重点检查]")
            # print(f"  > TV Loss          : {loss_tv.item():.6f} (平滑约束)")
            # print(f"  > Gradient Scale   : {guidance_opt.lambda_guidance} (SDS 权重)")
            # print(f"----------------------------------------------------------\n")
            

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if opt.save_process:
                if iteration % save_process_iter == 0 and len(process_view_points) > 0:
                    viewpoint_cam_p = process_view_points.pop(0)
                    render_p = render(viewpoint_cam_p, gaussians, pipe, background, test=True)
                    img_p = torch.clamp(render_p["render"], 0.0, 1.0) 
                    img_p = img_p.detach().cpu().permute(1,2,0).numpy()
                    img_p = (img_p * 255).round().astype('uint8')
                    pro_img_frames.append(img_p)  

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in testing_iterations):
                if save_video:
                    video_inference(iteration, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration < 3000 and iteration % opt.opacity_reset_interval == 0: #or (dataset._white_background and iteration == opt.densify_from_iter)
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene._model_path + "/chkpnt" + str(iteration) + ".pth")

    if opt.save_process:
        imageio.mimwrite(os.path.join(save_folder_proc, "video_rgb.mp4"), pro_img_frames, fps=30, quality=8)


def prepare_output_and_logger(args):    
    if not args._model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
            
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args._model_path = os.path.join("./output/", args.workspace + "_" + current_time)
        # args._model_path = os.path.join("./output/", args.workspace)
        
    # Set up output folder
    print("Output folder: {}".format(args._model_path))
    os.makedirs(args._model_path, exist_ok = True)

    # copy configs
    if args.opt_path is not None:
        os.system(' '.join(['cp', args.opt_path, os.path.join(args._model_path, 'config.yaml')]))

    with open(os.path.join(args._model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args._model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        save_folder = os.path.join(scene.args._model_path,"test_six_views/{}_iteration".format(iteration))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)  # makedirs 创建文件时如果路径不存在会创建这个路径
            print('test views is in :', save_folder)
        torch.cuda.empty_cache()
        config = ({'name': 'test', 'cameras' : scene.getTestCameras()})
        if config['cameras'] and len(config['cameras']) > 0:
            for idx, viewpoint in enumerate(config['cameras']):
                render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
                rgb, depth = render_out["render"],render_out["depth"]
                if depth is not None:
                    # 1. 制造一个 Mask，把背景(深度为0的部分)排除掉
                    # 注意：根据你的渲染器不同，背景可能是0也可能是极大值，这里假设是0
                    mask = depth > 0 
                    
                    # 只有当画面里有东西时才处理
                    if mask.sum() > 0:
                        # 2. 只从“有物体”的地方取最大最小值
                        d_valid = depth[mask]
                        d_min = d_valid.min()
                        d_max = d_valid.max()
                        
                        # 3. 归一化：(x - min) / (max - min)
                        # 这样最近的点变成0，最远的点变成1，拉满了整个 0-1 的区间
                        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
                        
                        # 4. 反转颜色 (论文常用：近处白，远处黑)
                        # 因为前面归一化后，最近点(d_min)是0(黑)，最远点(d_max)是1(白)
                        # 所以我们需要反过来：1 - x
                        depth_norm = 1.0 - depth_norm
                        
                        # 5. 把背景重新把抹黑
                        # 因为反转后，背景(原本比d_min还小)可能变成了大于1的数或者是乱七八糟的数
                        depth_norm[~mask] = 0.0
                        
                        # 保存
                        save_image(depth_norm, os.path.join(save_folder, "render_depth_{}.png".format(viewpoint.uid)))
                
                if depth is not None:
                    depth_norm = depth/depth.max()
                    save_image(depth_norm,os.path.join(save_folder,"render_depth_{}.png".format(viewpoint.uid)))

                image = torch.clamp(rgb, 0.0, 1.0)
                save_image(image,os.path.join(save_folder,"render_view_{}.png".format(viewpoint.uid)))
                if tb_writer:
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.uid), image[None], global_step=iteration)     
            print("\n[ITER {}] Eval Done!".format(iteration))
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def video_inference(iteration, scene : Scene, renderFunc, renderArgs):
    sharp = T.RandomAdjustSharpness(3, p=1.0)

    save_folder = os.path.join(scene.args._model_path,"videos/{}_iteration".format(iteration))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs 
        print('videos is in :', save_folder)
    torch.cuda.empty_cache()
    config = ({'name': 'test', 'cameras' : scene.getCircleVideoCameras()})
    if config['cameras'] and len(config['cameras']) > 0:
        img_frames = []
        depth_frames = []
        print("Generating Video using", len(config['cameras']), "different view points")
        for idx, viewpoint in enumerate(config['cameras']):
            render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
            rgb,depth = render_out["render"],render_out["depth"]
            if depth is not None:
                depth_norm = depth/depth.max()
                depths = torch.clamp(depth_norm, 0.0, 1.0) 
                depths = depths.detach().cpu().permute(1,2,0).numpy()
                depths = (depths * 255).round().astype('uint8')          
                depth_frames.append(depths)    
  
            image = torch.clamp(rgb, 0.0, 1.0) 
            image = image.detach().cpu().permute(1,2,0).numpy()
            image = (image * 255).round().astype('uint8')
            img_frames.append(image)    
            #save_image(image,os.path.join(save_folder,"lora_view_{}.jpg".format(viewpoint.uid)))   
        # Img to Numpy
        imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), img_frames, fps=30, quality=8)
        if len(depth_frames) > 0:
            imageio.mimwrite(os.path.join(save_folder, "video_depth_{}.mp4".format(iteration)), depth_frames, fps=30, quality=8)
        print("\n[ITER {}] Video Save Done!".format(iteration))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import yaml

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_ratio", type=int, default=5) # [2500,5000,7500,10000,12000]
    parser.add_argument("--save_ratio", type=int, default=2) # [10000,12000]
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # parser.add_argument("--device", type=str, default='cuda')

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gcp = GenerateCamParams(parser)
    gp = GuidanceParams(parser)

    args = parser.parse_args(sys.argv[1:])

    if args.opt is not None:
        with open(args.opt) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
        lp.load_yaml(opts.get('ModelParams', None))
        op.load_yaml(opts.get('OptimizationParams', None))
        pp.load_yaml(opts.get('PipelineParams', None))
        gcp.load_yaml(opts.get('GenerateCamParams', None))
        gp.load_yaml(opts.get('GuidanceParams', None))
        
        lp.opt_path = args.opt
        args.port = opts['port']
        args.save_video = opts.get('save_video', True)
        args.seed = opts.get('seed', 0)
        args.device = opts.get('device', 'cuda')

        # override device
        gp.g_device = args.device
        lp.data_device = args.device
        gcp.device = args.device

    # save iterations
    test_iter = [1] + [k * op.iterations // args.test_ratio for k in range(1, args.test_ratio)] + [op.iterations]
    args.test_iterations = test_iter

    save_iter = [k * op.iterations // args.save_ratio for k in range(1, args.save_ratio)] + [op.iterations]
    args.save_iterations = save_iter

    print('Test iter:', args.test_iterations)
    print('Save iter:', args.save_iterations)

    print("Optimizing " + lp._model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp, op, pp, gcp, gp, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.save_video)

    # All done
    print("\nTraining complete.")
