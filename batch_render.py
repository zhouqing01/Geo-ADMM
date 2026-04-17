"""
批量渲染脚本 - 渲染 lucid_simple 下所有子文件夹的 model.ply

使用方法:
    conda activate Lucid
    cd /workspace/LucidDreamer
    python batch_render.py

输出结构:
    lucid_simple/
    ├── A_cactus_with_pink_flowers/
    │   ├── model.ply
    │   └── rendered_views/    <-- 新建
    │       ├── 000.png
    │       ├── 001.png
    │       └── ...
    ├── A_rainbow_colored_umbrella/
    │   ├── model.ply
    │   └── rendered_views/
    └── ...
"""
import os
import sys
sys.path.insert(0, '/workspace/LucidDreamer')

import torch
import numpy as np
import math
import imageio
import argparse
from tqdm import tqdm
from glob import glob
import trimesh

# LucidDreamer imports
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class SimplePipe:
    def __init__(self):
        self.compute_cov3D_python = False
        self.convert_SHs_python = False

class LucidCamera:
    def __init__(self, R, T, FoVx, FoVy, H, W, znear=0.01, zfar=100.0, device="cuda"):
        self.FoVy = FoVy
        self.FoVx = FoVx
        self.image_height = H
        self.image_width = W
        self.znear = znear
        self.zfar = zfar
        
        RT = torch.tensor(getWorld2View2(R, T))
        self.world_view_transform = RT.transpose(0, 1).to(device)
        self.projection_matrix = getProjectionMatrix(
            znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy
        ).transpose(0, 1).to(device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def get_meshrender_pose(v):
    f = v / np.linalg.norm(v)
    u = np.array([0, 0, 1]) if np.abs(np.dot(f, [0, 0, 1])) < 0.99 else np.array([0, 1, 0])
    r = np.cross(u, -v)
    r = r / np.linalg.norm(r)
    u_ = np.cross(-v, r)
    u_ = u_ / np.linalg.norm(u_)
    R_c2w = np.stack([-r, u_, f], axis=1)
    return R_c2w, v

def convert_pose_to_view_matrix(R_c2w, center):
    R_w2c = R_c2w.T
    R_gl_to_cv = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float32)
    R_final = R_gl_to_cv @ R_w2c
    T_final = -np.dot(R_final, center)
    return R_final.T, T_final

def clean_gaussians(gaussians, device):
    xyz = gaussians._xyz
    n_points = xyz.shape[0]
    mask = torch.isfinite(xyz).all(dim=1)
    
    for attr in ['_features_dc', '_opacity', '_scaling', '_rotation']:
        if hasattr(gaussians, attr) and getattr(gaussians, attr) is not None:
            val = getattr(gaussians, attr)
            if val.dim() > 1:
                mask &= torch.isfinite(val.reshape(n_points, -1)).all(dim=1)
            else:
                mask &= torch.isfinite(val)

    valid_count = mask.sum().item()
    removed_count = n_points - valid_count
    
    if removed_count > 0:
        gaussians._xyz = gaussians._xyz[mask]
        for attr in ['_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation']:
            if hasattr(gaussians, attr) and getattr(gaussians, attr) is not None:
                setattr(gaussians, attr, getattr(gaussians, attr)[mask])
    
    for attr in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
        if hasattr(gaussians, attr) and getattr(gaussians, attr) is not None:
            setattr(gaussians, attr, getattr(gaussians, attr).to(device))
    
    return gaussians, valid_count, removed_count

def normalize_gaussians(gaussians):
    xyz = gaussians._xyz
    shift = xyz.mean(dim=0)
    xyz = xyz - shift
    max_dist = xyz.abs().max()
    if max_dist > 0:
        xyz = xyz / max_dist
        gaussians._scaling = gaussians._scaling - torch.log(max_dist)
    gaussians._xyz = xyz
    return gaussians

def render_single_ply(ply_path, output_dir, subdivisions=2, radius=2.2, device="cuda"):
    """渲染单个 PLY 文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    H, W = 512, 512
    fov_y = np.pi / 3.0
    fov_x = 2 * math.atan(math.tan(fov_y / 2) * (W / H))
    
    icosphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    cam_positions = icosphere.vertices
    num_views = len(cam_positions)
    
    try:
        gaussians = GaussianModel(sh_degree=0)
        gaussians.load_ply(ply_path)
        total_points = gaussians._xyz.shape[0]
        
        gaussians, valid_count, removed_count = clean_gaussians(gaussians, device)
        
        if valid_count == 0:
            raise ValueError("No valid points!")
        
        gaussians = normalize_gaussians(gaussians)
        
    except Exception as e:
        print(f"    [Error] {e}")
        # 生成黑色占位图
        for idx in range(num_views):
            imageio.imwrite(f'{output_dir}/{idx:03d}.png', np.zeros((H, W, 3), dtype=np.uint8))
        return False, 0, 0
    
    pipe = SimplePipe()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    
    for idx, v in enumerate(cam_positions):
        R_c2w, center = get_meshrender_pose(v)
        R_gs, T_gs = convert_pose_to_view_matrix(R_c2w, center)
        camera = LucidCamera(R_gs, T_gs, fov_x, fov_y, H, W, device=device)
        
        with torch.no_grad():
            output = render(camera, gaussians, pipe, background)
            rendered = output["render"]
            depth = output["depth"].squeeze()
            alpha = output["alpha"].squeeze()
            
        max_a = alpha.max().item()
        print(f"检测到 Alpha 最大值: {max_a:.4f}")
        
        if max_a > 1e-5:
            # 动态阈值：取最大不透明度的 50%，剪掉边缘浮点
            thresh = max_a * 0.8 
            mask = alpha > thresh
            
            obj_depths = depth[mask]
            if obj_depths.numel() > 0:
                # 自动锁定物体深度区间 (5% - 95%)
                v_min = torch.quantile(obj_depths, 0.05)
                v_max = torch.quantile(obj_depths, 0.95)
                
                # 线性映射
                res = (depth - v_min) / (v_max - v_min + 1e-7)
                res = torch.clamp(res, 0.0, 1.0)
                
                # 核心：使用归一化 Alpha 进行权重消融，彻底抹除白边
                alpha_s = torch.clamp((alpha - thresh) / (max_a - thresh + 1e-7), 0, 1)
                res = res * alpha_s 
                res[~mask] = 0
            else:
                res = torch.zeros_like(depth)
        else:
            print("警告：画面中未发现物体！")
            res = torch.zeros_like(depth)

        img = (res.cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(f'{output_dir}/{idx:03d}_depth.png', img)
        
        rendered_np = rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        rendered_np = (rendered_np * 255).astype(np.uint8)
        imageio.imwrite(f'{output_dir}/{idx:03d}.png', rendered_np)
    
    return True, total_points, valid_count

def main():
    parser = argparse.ArgumentParser(description="Batch render all PLY files in lucid_simple")
    parser.add_argument('--input_dir', type=str, 
                        default='/workspace/LucidDreamer/output_original_t3/lucid_simple',
                        help='Directory containing subdirectories with model.ply')
    parser.add_argument('--output_folder', type=str, default='rendered_views',
                        help='Name of output folder in each subdirectory')
    parser.add_argument('--subdivisions', type=int, default=2,
                        help='Icosphere subdivisions (0=12, 1=42, 2=162 views)')
    parser.add_argument('--radius', type=float, default=2.2, help='Camera distance')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip if output folder already exists')
    args = parser.parse_args()
    
    # 查找所有 model.ply 文件
    # ply_files = sorted(glob(os.path.join(args.input_dir, '*/model.ply')))
    
    ply_files = ['/workspace/LucidDreamer/output/bunny_2026-01-25_16-10-03/point_cloud/iteration_5000/point_cloud.ply']
    
    if not ply_files:
        print(f"No model.ply files found in {args.input_dir}")
        return
    
    print(f"Found {len(ply_files)} PLY files to render")
    print(f"Output folder name: {args.output_folder}")
    print(f"Views per model: {len(trimesh.creation.icosphere(subdivisions=args.subdivisions).vertices)}")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for i, ply_path in enumerate(ply_files):
        # 获取子文件夹名称
        subdir = os.path.dirname(ply_path)
        folder_name = os.path.basename(subdir)
        output_dir = os.path.join(subdir, args.output_folder)
        
        print(f"\n[{i+1}/{len(ply_files)}] {folder_name}")
        
        # 检查是否跳过
        if args.skip_existing and os.path.exists(output_dir):
            num_images = len(glob(os.path.join(output_dir, '*.png')))
            if num_images > 0:
                print(f"    Skipping (already has {num_images} images)")
                success_count += 1
                continue
        
        # 渲染
        success, total, valid = render_single_ply(
            ply_path=ply_path,
            output_dir=output_dir,
            subdivisions=args.subdivisions,
            radius=args.radius
        )
        
        if success:
            print(f"    ✓ Done: {valid}/{total} valid points, saved to {args.output_folder}/")
            success_count += 1
        else:
            print(f"    ✗ Failed")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"Completed: {success_count} success, {fail_count} failed")

if __name__ == "__main__":
    main()
