# ================= [Super Monkey Patch] =================
# 必须放在最开头！用于一次性解决 ImageReward 和新版 transformers 的所有冲突
import os
import re  # 新增正则库
import torch
import transformers.modeling_utils

# 补丁 1: apply_chunking_to_forward
if not hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        return forward_fn(*input_tensors)
    transformers.modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward

# 补丁 2: find_pruneable_heads_and_indices
if not hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        return set(), None
    transformers.modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

# 补丁 3: prune_linear_layer
if not hasattr(transformers.modeling_utils, "prune_linear_layer"):
    def prune_linear_layer(layer, index, dim=0):
        return layer
    transformers.modeling_utils.prune_linear_layer = prune_linear_layer

print("[PATCH] 已注入 transformers 兼容性补丁。")
# ================= [Patch End] =================

import os
import glob
import yaml
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
ROOT_DIR = "/workspace/LucidDreamer/output"
TARGET_SUBPATH = "test_six_views/5000_iteration"
OUTPUT_CSV = "batch_evaluation_results_base.csv"
SUMMARY_CSV = "batch_evaluation_best_summary_base.csv" # 新增：冠军榜文件名
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageReward 权重文件的绝对路径
IR_WEIGHTS_PATH = "/workspace/zai-org/ImageReward/ImageReward.pt"
IR_CONFIG_PATH = "/workspace/zai-org/ImageReward/med_config.json"
# ===========================================

# 全局标记
HAS_CLIP = False
HAS_IR = False

# 1. 尝试加载 CLIP
try:
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    print("[ENV] 当前环境不支持 transformers/CLIP")

# 2. 尝试导入 ImageReward 类 (手动加载模式)
try:
    # 注意：这里尝试导入核心模型类，而不是整个库，绕过 init 里的检查
    from ImageReward.ImageReward import ImageReward as ImageRewardModel
    HAS_IR = True
except ImportError:
    # 如果找不到，尝试退回到标准导入 (兼容老环境)
    try:
        import ImageReward as RM
        HAS_IR = True
        print("[ENV] 使用标准 ImageReward 导入")
    except ImportError:
        print("[ENV] 当前环境未安装 ImageReward")
except Exception as e:
    print(f"[ENV] ImageReward 类加载失败: {e}")

def load_models():
    # === Global 声明 ===
    global HAS_CLIP, HAS_IR
    
    clip_model = None
    clip_processor = None
    reward_model = None

    # === 加载 CLIP ===
    if HAS_CLIP:
        print("[INIT] 正在加载 CLIP 模型...")
        # clip_id = "/workspace/openai/clip-vit-large-patch14"
        clip_id = "/workspace/openai/clip-vit-base-patch32"
        try:
            clip_model = CLIPModel.from_pretrained(clip_id).to(DEVICE)
            clip_processor = CLIPProcessor.from_pretrained(clip_id)
        except Exception as e:
            print(f"[WARN] CLIP 加载失败: {e}")
            HAS_CLIP = False
    
    # === 加载 ImageReward (手动离线模式) ===
    if HAS_IR:
        print(f"[INIT] 正在加载 ImageReward...")
        print(f"       权重: {IR_WEIGHTS_PATH}")
        print(f"       配置: {IR_CONFIG_PATH}")
        
        # 检查两个文件是否都存在
        if os.path.exists(IR_WEIGHTS_PATH) and os.path.exists(IR_CONFIG_PATH):
            try:
                # === [修改] 直接指定我们准备好的配置文件路径 ===
                reward_model = ImageRewardModel(device=DEVICE, med_config=IR_CONFIG_PATH)
                
                # 手动加载权重
                state_dict = torch.load(IR_WEIGHTS_PATH, map_location=DEVICE)
                reward_model.load_state_dict(state_dict, strict=False)
                reward_model.to(DEVICE)
                reward_model.eval()
                
                print("[SUCCESS] ImageReward 模型加载成功！")
                
            except Exception as e:
                print(f"[ERROR] 手动加载 ImageReward 失败: {e}")
                HAS_IR = False
        else:
            print(f"[ERROR] 离线文件缺失！")
            if not os.path.exists(IR_WEIGHTS_PATH):
                print(f"  ❌ 缺少权重文件: {IR_WEIGHTS_PATH}")
            if not os.path.exists(IR_CONFIG_PATH):
                print(f"  ❌ 缺少配置文件: {IR_CONFIG_PATH}")
            HAS_IR = False

    return clip_model, clip_processor, reward_model

def get_prompt_from_yaml(exp_path):
    yaml_path = os.path.join(exp_path, "config.yaml")
    if not os.path.exists(yaml_path): return None
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config and 'GuidanceParams' in config:
                return config['GuidanceParams'].get('text', None)
    except:
        pass
    return None

def calculate_consistency(img_embeds):
    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
    sim_matrix = torch.matmul(img_embeds, img_embeds.t())
    n = sim_matrix.shape[0]
    if n <= 1: return 0.0
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    return sim_matrix[mask].mean().item()

def get_base_class_name(exp_name):
    # 逻辑：匹配 _YYYY-MM-DD 之前的所有字符作为类别名
    # 例如: zombie_joker_2025-12-30... -> zombie_joker
    match = re.search(r"^(.*)_\d{4}-\d{2}-\d{2}", exp_name)
    if match:
        return match.group(1)
    return exp_name # 如果没匹配到日期，就用原名

def main():
    if not HAS_CLIP and not HAS_IR:
        print("[ERROR] 无法进行评估。")
        return

    clip_model, clip_processor, reward_model = load_models()
    
    # 读取旧数据
    existing_data = {}
    csv_path = os.path.join(ROOT_DIR, OUTPUT_CSV)
    if os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            existing_data = df_old.set_index("Experiment").to_dict('index')
        except: pass

    exp_folders = sorted([f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))])
    results_list = []

    # === 遍历评估 ===
    for exp_name in tqdm(exp_folders, desc="Evaluating"):
        exp_path = os.path.join(ROOT_DIR, exp_name)
        img_dir = os.path.join(exp_path, TARGET_SUBPATH)
        
        current_data = existing_data.get(exp_name, {})
        current_data["Experiment"] = exp_name
        
        need_run_clip = HAS_CLIP and pd.isna(current_data.get("CLIP_Score"))
        need_run_ir = HAS_IR and pd.isna(current_data.get("ImageReward"))
        
        if not (need_run_clip or need_run_ir):
            if os.path.exists(img_dir): results_list.append(current_data)
            continue

        if not os.path.exists(img_dir): continue
        image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not image_paths: continue

        prompt = get_prompt_from_yaml(exp_path)
        if not prompt: continue
        
        current_data["Prompt"] = prompt
        current_data["Image_Count"] = len(image_paths)

        if need_run_clip and clip_model:
            try:
                images = [Image.open(p).convert("RGB") for p in image_paths]
                inputs = clip_processor(text=[prompt[:77]], images=images, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    img_embeds = outputs.image_embeds
                    text_embeds = outputs.text_embeds
                
                img_embeds_norm = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds_norm = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                
                current_data["CLIP_Score"] = (img_embeds_norm @ text_embeds_norm.t()).squeeze().mean().item() * 100
                current_data["Consistency"] = calculate_consistency(img_embeds)
            except Exception as e:
                print(f"[ERR] CLIP: {e}")

        if need_run_ir and reward_model:
            try:
                rewards = []
                with torch.no_grad():
                    for img_path in image_paths:
                        score = reward_model.score(prompt, img_path)
                        rewards.append(score)
                current_data["ImageReward"] = sum(rewards) / len(rewards)
            except Exception as e:
                print(f"[ERR] IR: {e}")

        results_list.append(current_data)

    # === 保存并统计最佳结果 ===
    if results_list:
        df = pd.DataFrame(results_list)
        
        # 1. 保存所有数据
        cols = ["Experiment", "Prompt", "CLIP_Score", "ImageReward", "Consistency", "Image_Count"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] 表格已更新: {csv_path}")

        # 2. 统计每类最佳结果 (Best of Class)
        print("\n" + "="*50)
        print("🏆 各类别最佳模型统计 (Based on ImageReward) 🏆")
        print("="*50)

        # 提取基础类名
        df['Class_Name'] = df['Experiment'].apply(get_base_class_name)
        
        best_rows = []
        
        # 按类分组
        grouped = df.groupby('Class_Name')
        for name, group in grouped:
            best_idx = None
            
            # 优先使用 ImageReward 选最好的
            if 'ImageReward' in group.columns and not group['ImageReward'].isna().all():
                best_idx = group['ImageReward'].idxmax()
            # 其次使用 CLIP Score
            elif 'CLIP_Score' in group.columns and not group['CLIP_Score'].isna().all():
                best_idx = group['CLIP_Score'].idxmax()
            
            if best_idx is not None:
                best_row = group.loc[best_idx]
                best_rows.append(best_row)
                
                # 打印到控制台
                ir_val = f"{best_row.get('ImageReward', 0):.4f}" if not pd.isna(best_row.get('ImageReward')) else "N/A"
                clip_val = f"{best_row.get('CLIP_Score', 0):.2f}" if not pd.isna(best_row.get('CLIP_Score')) else "N/A"
                
                print(f"📌 类别: {name}")
                print(f"   最佳实验: {best_row['Experiment']}")
                print(f"   得分: IR={ir_val}, CLIP={clip_val}")
                print("-" * 30)

        # 3. 保存最佳结果到单独 CSV
        if best_rows:
            df_best = pd.DataFrame(best_rows)
            # 清理一下临时的 Class_Name 列
            output_cols = [c for c in cols if c in df_best.columns]
            summary_path = os.path.join(ROOT_DIR, SUMMARY_CSV)
            df_best[output_cols].to_csv(summary_path, index=False)
            print(f"[INFO] 最佳结果汇总已保存至: {summary_path}")
            print("="*50 + "\n")

if __name__ == "__main__":
    main()