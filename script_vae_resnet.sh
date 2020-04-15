#!/bin/bash 
# activate the virtual environment

echo "Images:25 Run 1"
python finetune_vae_2classes.py -m vae_c23_latent128_resnet3blocks_2classes_update_last -r 1 -i 25 &&

echo "Images:25 Run 2"
python finetune_vae_2classes.py -m vae_c23_latent128_resnet3blocks_2classes_update_last -r 2 -i 25 &&

echo "Images:25 Run 3"
python finetune_vae_2classes.py -m vae_c23_latent128_resnet3blocks_2classes_update_last -r 3 -i 25  &&

echo "Images:25 Run 4"
python finetune_vae_2classes.py -m vae_c23_latent128_resnet3blocks_2classes_update_last -r 4 -i 25  &&

echo "Images:25 Run 5"
python finetune_vae_2classes.py -m vae_c23_latent128_resnet3blocks_2classes_update_last -r 5 -i 25  &&

echo "Images:25 Run 6"
python finetune_vae_2classes.py -m vae_c23_latent128_resnet3blocks_2classes_update_last -r 6 -i 25  &&

echo "Images:25 Run 7"
python finetune_vae_2classes.py -m vae_c23_latent128_resnet3blocks_2classes_update_last -r 7 -i 25  &&

echo "Images:25 Run 8"
python finetune_vae_2classes.py -m vae_c23_latent128_resnet3blocks_2classes_update_last -r 8 -i 25  &&

echo "Images:25 Run 9"
python finetune_vae_2classes.py -m vae_c23_latent128_resnet3blocks_2classes_update_last -r 9 -i 25  &&

echo "Images:25 Run 10"
python finetune_vae_2classes.py -m vae_c23_latent128_resnet3blocks_2classes_update_last -r 10 -i 25
#python calc_mean.py