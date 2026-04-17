# import torch
# @torch.no_grad()
# def overlap_laplacian_prox(
#     xyz,                  # [M, 3] visible gaussian centers
#     radii,                # [M] projected radius (pixel-space)
#     visibility_filter,    # [M] bool
#     lambda_overlap=0.1,
#     iters=5
# ):
#     """
#     Gaussian overlap MBIR prox step (z-update)

#     Returns:
#         z_xyz: [M, 3] smoothed centers
#     """

#     # only visible gaussians
#     x = xyz[visibility_filter]          # [M, 3]
#     r = radii[visibility_filter]        # [M]

#     z = x.clone()

#     for _ in range(iters):
#         # pairwise overlap weight (cheap approximation)
#         # w_ij = exp( -||xi-xj||^2 / (ri+rj)^2 )
#         diff = z[:, None, :] - z[None, :, :]      # [M, M, 3]
#         dist2 = (diff ** 2).sum(-1)                # [M, M]

#         scale = (r[:, None] + r[None, :]).clamp(min=1e-6)
#         w = torch.exp(-dist2 / (scale ** 2))

#         # normalize
#         w = w / (w.sum(dim=1, keepdim=True) + 1e-6)

#         # Laplacian smoothing
#         z = (1 - lambda_overlap) * z + lambda_overlap * (w @ z)

#     z_full = xyz.clone()
#     z_full[visibility_filter] = z
#     return z_full

import torch

@torch.no_grad()
def overlap_laplacian_prox(
    xyz,                  # [M, 3]
    radii,                # [M]
    visibility_filter,    # [M] bool
    lambda_overlap=0.1,
    iters=5,
    sample_k=1024,        # 新增参数：每次只与 K 个随机点计算相互作用
    chunk_size=4096       # 分块处理以防 [M, K] 矩阵依然太大
):
    x = xyz[visibility_filter]      # [M, 3]
    r = radii[visibility_filter]    # [M]
    
    M = x.shape[0]
    z = x.clone()

    for _ in range(iters):
        # 1. 随机采样 sample_k 个点作为“邻居”参考
        # 如果 M 很小，就直接用全部；如果很大，就采样
        if M > sample_k:
            idx = torch.randperm(M, device=z.device)[:sample_k]
            z_target = z[idx]       # [K, 3]
            r_target = r[idx]       # [K]
        else:
            z_target = z
            r_target = r

        # 2. 初始化更新量的累加器
        # 我们需要计算 w @ z_target，结果形状是 [M, 3]
        # 公式: z_new = (sum(w_ij * z_j) / sum(w_ij))
        
        weighted_pos_sum = torch.zeros_like(z) # [M, 3]
        weight_sum = torch.zeros((M, 1), device=z.device) # [M, 1]

        # 3. 分块计算 (Chunking) 以避免 OOM
        # 此时显存占用为 O(chunk_size * sample_k)
        for i in range(0, M, chunk_size):
            end = min(i + chunk_size, M)
            
            # 当前块的源点 [B, 3]
            z_src_chunk = z[i:end] 
            r_src_chunk = r[i:end]
            
            # 计算距离 [B, K]
            # diff: [B, 1, 3] - [1, K, 3] -> [B, K, 3]
            diff = z_src_chunk[:, None, :] - z_target[None, :, :]
            dist2 = (diff ** 2).sum(-1) # [B, K]
            
            # 计算 Scale [B, K]
            scale = (r_src_chunk[:, None] + r_target[None, :]).clamp(min=1e-6)
            
            # 计算权重
            w = torch.exp(-dist2 / (scale ** 2)) # [B, K]
            
            # 累加分子和分母
            # w @ z_target -> [B, K] @ [K, 3] -> [B, 3]
            weighted_pos_sum[i:end] = torch.mm(w, z_target)
            weight_sum[i:end] = w.sum(dim=1, keepdim=True)

        # 4. 归一化并更新
        # z_smooth = weighted_pos_sum / weight_sum
        z_smooth = weighted_pos_sum / (weight_sum + 1e-6)
        
        z = (1 - lambda_overlap) * z + lambda_overlap * z_smooth

    z_full = xyz.clone()
    z_full[visibility_filter] = z
    return z_full