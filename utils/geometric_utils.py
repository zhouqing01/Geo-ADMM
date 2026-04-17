import torch

def get_knn_chunked(x, k=20, chunk_size=1000):
    """
    纯 PyTorch 实现的分块 KNN，避免 N^2 显存爆炸。
    适合 N < 200k 的情况。如果显存依然不够，请减小 chunk_size (如 500)。
    
    Args:
        x: [N, 3] 点云坐标
        k: 邻居数量
        chunk_size: 每次计算距离的批次大小
    Returns:
        indices: [N, k] 邻居索引 (不包含自己)
    """
    N = x.shape[0]
    indices_list = []
    
    # 预计算 x 的平方和: ||x||^2 [N, 1]
    x_sq = (x**2).sum(dim=1, keepdim=True)
    
    # 转置 x 以便矩阵乘法
    x_t = x.t()
    
    # 分块处理查询点
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        
        # 取出一个 chunk 的查询点: [chunk_size, 3]
        x_chunk = x[i:end]
        x_sq_chunk = x_sq[i:end]
        
        # 计算距离矩阵 D^2 = ||x||^2 + ||y||^2 - 2xy
        # [chunk, 1] + [1, N] - [chunk, N] = [chunk, N]
        dist_sq = x_sq_chunk + x_sq.t() - 2 * torch.mm(x_chunk, x_t)
        
        # 获取最小的 k+1 个距离 (包含自己，因为距离为0)
        # largest=False 表示取最小
        _, idx = dist_sq.topk(k=k + 1, largest=False, dim=1)
        
        # 去掉第一个 (自己)
        indices_list.append(idx[:, 1:])
        
    return torch.cat(indices_list, dim=0)

def compute_laplacian(x, k=20):
    """
    计算点云的拉普拉斯向量 (近似平均曲率流方向 Mean Curvature Flow)。
    L = Mean(Neighbors) - Current_Position
    
    物理含义：
    这个向量指向局部几何的“中心”。沿着这个方向移动点，会使凸起的部分收缩，
    凹陷的部分填平，从而实现去噪和几何正则化。
    """
    with torch.no_grad(): # 几何拓扑计算不需要梯度回传，只作为 Target
        N = x.shape[0]
        
        # 1. 寻找邻居 (耗时操作)
        # 如果点数非常多，可以适当减小 k 以加速 (如 k=10)
        knn_idx = get_knn_chunked(x, k=k, chunk_size=500) # [N, K]
        
        # 2. 获取邻居坐标 [N, K, 3]
        # 使用 gather 收集邻居坐标
        # neighbors = x[knn_idx] 
        # 但为了节省显存，我们不显式构建 [N, K, 3] 张量
        # 而是利用 reshape 技巧
        
        neighbors = x[knn_idx.view(-1)].view(N, k, 3)
        
        # 3. 计算局部质心 (Local Centroid)
        centroid = neighbors.mean(dim=1) # [N, 3]
        
        # 4. 拉普拉斯向量 (指向质心)
        laplacian = centroid - x # [N, 3]
        
    return laplacian

def solve_admm_z_update(curr_theta, u_dual, rho, conflict_map=None, flow_alpha=0.1, k=10):
    """
    模拟 FlowDreamer / Duan's Flow 的 ADMM Z-Step (几何投影)。
    
    Args:
        curr_theta: [N, 3] 当前高斯球位置
        u_dual: [N, 3] 对偶变量
        rho: ADMM 惩罚参数
        conflict_map: [N, 1] JSM 权重 (0~1)。
                      1.0 = 共识区 (保留细节)
                      0.0 = 冲突区 (Janus面/Outlier，需要强力收缩)
        flow_alpha: 控制平滑力度的基础系数
        k: KNN 邻居数
    
    Returns:
        z_new: [N, 3] 更新后的几何辅助变量
    """
    # 0. 维度检查 (防止 crash)
    if curr_theta.shape[0] != u_dual.shape[0]:
        # 如果维度不匹配，说明发生了 densify/prune，Z step 无法进行
        # 直接返回 curr_theta，相当于跳过这一步几何约束
        return curr_theta

    # 1. 计算 ADMM 目标位置 T = Theta + U
    target = curr_theta + u_dual
    
    # 2. 计算拉普拉斯收缩向量 (Geometric Shrinkage Vector)
    # 这个向量指向“更平滑/收缩”的位置
    laplacian = compute_laplacian(target, k=k)
    
    # 3. 计算自适应平滑强度 (Adaptive Shrinkage Strength)
    # 逻辑：
    # 如果 conflict_map 接近 1 (无冲突)，我们希望保留原状 -> strength 小
    # 如果 conflict_map 接近 0 (冲突大)，我们希望强力平滑/收缩 -> strength 大
    
    if conflict_map is None:
        # 如果没有冲突图，就做全局轻微平滑
        shrinkage_strength = flow_alpha
    else:
        # 确保 conflict_map 维度匹配
        if conflict_map.shape[0] != target.shape[0]:
             # 如果 conflict map 维度对不上(比如是用旧点云计算的)，降级为全局平滑
             shrinkage_strength = flow_alpha
        else:
            # 反转权重：冲突越大(w_jsm小)，平滑力度越大
            # 增加一个基数 0.05 保证即使在好区域也有微弱的正则化（防止飞点）
            shrinkage_strength = flow_alpha * (1.0 - conflict_map + 0.05)
    
    # 4. 执行显式更新 (Explicit Update)
    # Z = Target + Step * Strength * Laplacian
    # 这里的步长由 ADMM 的 rho 决定。rho 越大，说明我们要越紧地贴合 Target，
    # 因此几何流的修正幅度应该相对变小 (1/rho)。
    
    step_size = 1.0 / (rho + 1e-5) 
    
    z_new = target + step_size * shrinkage_strength * laplacian
    
    return z_new