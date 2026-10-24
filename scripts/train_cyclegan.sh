set -ex
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout
python train.py --dataroot ./datasets/grumpifycat --name grumpifycat_cyclegan --model cycle_gan --pool_size 50 --no_dropout --use_wandb
python train.py --dataroot ./datasets/grumpifycat --name grumpify_cat --model cycle_gan --display_id 0
