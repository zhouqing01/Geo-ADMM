export CUDA_VISIBLE_DEVICES="0" # 0,1,2,3
# python train_mbir.py --opt './configs/axe.yaml'
# python train_mbir.py --opt './configs/bagel.yaml'
# python train_mbir.py --opt './configs/cat_armor.yaml'
# python train_mbir.py --opt './configs/crown.yaml'
# python train_mbir.py --opt './configs/football_helmet.yaml'
# python train_mbir.py --opt './configs/hamburger.yaml'
# python train_mbir.py --opt './configs/ts_lora.yaml'
# python train_mbir.py --opt './configs/white_hair_ironman.yaml'
# python train_mbir.py --opt './configs/zombie_joker.yaml'
python train_mbir.py --opt './configs/puppy.yaml'

python train_mbir.py --opt './configs/yaml_configs_new/extra/Schnauzer.yaml'

python train.py --opt './configs/yaml_configs_new/extra/bunny.yaml'
# python train_mbir_3d_admm.py --opt './configs/white_hair_ironman.yaml'
python train_dpm.py --opt './configs/white_hair_ironman.yaml'
python train_dpm.py --opt './configs/zombie_joker.yaml'

CUDA_VISIBLE_DEVICES="7" python train_mbir.py --opt './configs/yaml_configs_new/extra/bunny.yaml'

CUDA_VISIBLE_DEVICES="1" python train_mbir.py --opt './configs/yaml_configs_new/01_characters/charizard.yaml'
CUDA_VISIBLE_DEVICES="2" python train_mbir.py --opt './configs/yaml_configs_new/01_characters/mario.yaml'