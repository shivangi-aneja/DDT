#!/bin/bash 
# activate the virtual environment

echo "Images:25 Run 1"
python3 finetune_ddt.py -m ddt_c23_latent128_resnet3blocks_2classes_update_last -r 1  --train_mode train &&

echo "Images:25 Run 2"
python3 finetune_ddt.py -m ddt_c23_latent128_resnet3blocks_2classes_update_last -r 2  --train_mode train &&

echo "Images:25 Run 3"
python3 finetune_ddt.py -m ddt_c23_latent128_resnet3blocks_2classes_update_last -r 3   --train_mode train &&

echo "Images:25 Run 4"
python3 finetune_ddt.py -m ddt_c23_latent128_resnet3blocks_2classes_update_last -r 4   --train_mode train &&

echo "Images:25 Run 5"
python3 finetune_ddt.py -m ddt_c23_latent128_resnet3blocks_2classes_update_last -r 5   --train_mode train &&

echo "Images:25 Run 6"
python3 finetune_ddt.py -m ddt_c23_latent128_resnet3blocks_2classes_update_last -r 6   --train_mode train &&

echo "Images:25 Run 7"
python3 finetune_ddt.py -m ddt_c23_latent128_resnet3blocks_2classes_update_last -r 7   --train_mode train &&

echo "Images:25 Run 8"
python3 finetune_ddt.py -m ddt_c23_latent128_resnet3blocks_2classes_update_last -r 8   --train_mode train &&

echo "Images:25 Run 9"
python3 finetune_ddt.py -m ddt_c23_latent128_resnet3blocks_2classes_update_last -r 9   --train_mode train &&

echo "Images:25 Run 10"
python3 finetune_ddt.py -m ddt_c23_latent128_resnet3blocks_2classes_update_last -r 10  --train_mode train