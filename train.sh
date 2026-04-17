export CUDA_VISIBLE_DEVICES="1" # 0,1,2,3
# python train.py --opt './configs/axe.yaml'
# python train.py --opt './configs/bagel.yaml'
# python train.py --opt './configs/cat_armor.yaml'
# python train.py --opt './configs/crown.yaml'
# python train.py --opt './configs/football_helmet.yaml'
python train.py --opt './configs/hamburger.yaml'
# python train.py --opt './configs/ts_lora.yaml'
# python train.py --opt './configs/white_hair_ironman.yaml'
# python train.py --opt './configs/zombie_joker.yaml'

python multi_gpu_runner.py --prompts /workspace/LucidDreamer/prompt/simple.txt --config /workspace/LucidDreamer/configs/config_simple.yaml

python multi_gpu_runner.py --prompts /workspace/LucidDreamer/prompt/simple_surround.txt --config /workspace/LucidDreamer/configs/config_simple_surr.yaml

python multi_gpu_runner.py --prompts /workspace/LucidDreamer/prompt/multi.txt --config /workspace/LucidDreamer/configs/config_multi.yaml


# 生成的ply转图片
# --subdivisions 2 :162张
python batch_render.py --input_dir /workspace/LucidDreamer/output_original_t3/lucid_simple --subdivisions 2
python batch_render.py --input_dir /workspace/LucidDreamer/output_original_t3/lucid_simple_surround --subdivisions 2
python batch_render.py --input_dir /workspace/LucidDreamer/output_original_t3/lucid_multi --subdivisions 2 --skip_existing

python batch_render.py --input_dir /workspace/LucidDreamer/output_mbir/mbir_single --subdivisions 2
python batch_render.py --input_dir /workspace/LucidDreamer/output_original_t3/lucid_simple_surround --subdivisions 2
python batch_render.py --input_dir /workspace/LucidDreamer/output_original_t3/lucid_multi --subdivisions 2 --skip_existing

python batch_render.py --input_dir /workspace/LucidDreamer/output/bunny_2026-01-25_16-07-32/point_cloud/iteration_5000 --output_folder /workspace/LucidDreamer/output/bunny_2026-01-25_16-07-32/process --subdivisions 2

python batch_render.py --output_folder /workspace/LucidDreamer/output/bunny_2026-01-25_16-07-32/process --subdivisions 2