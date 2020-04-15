#!/bin/bash 

# activate the virtual environment
source ../env/venv/bin/activate

echo "Images:5 Run 1"
python finetune_forensic_transfer.py -m ft_c23_latent128_resnet3blocks_2classes_update_last -r 1 -i 5 &&

echo "Images:5 Run 2"
python finetune_forensic_transfer.py -m ft_c23_latent128_resnet3blocks_2classes_update_last -r 2 -i 5 &&

echo "Images:5 Run 3"
python finetune_forensic_transfer.py -m ft_c23_latent128_resnet3blocks_2classes_update_last -r 3 -i 5  &&

echo "Images:5 Run 4"
python finetune_forensic_transfer.py -m ft_c23_latent128_resnet3blocks_2classes_update_last -r 4 -i 5  &&

echo "Images:5 Run 5"
python finetune_forensic_transfer.py -m ft_c23_latent128_resnet3blocks_2classes_update_last -r 5 -i 5  &&

echo "Images:5 Run 6"
python finetune_forensic_transfer.py -m ft_c23_latent128_resnet3blocks_2classes_update_last -r 6 -i 5  &&

echo "Images:5 Run 7"
python finetune_forensic_transfer.py -m ft_c23_latent128_resnet3blocks_2classes_update_last -r 7 -i 5  &&

echo "Images:5 Run 8"
python finetune_forensic_transfer.py -m ft_c23_latent128_resnet3blocks_2classes_update_last -r 8 -i 5  &&

echo "Images:5 Run 9"
python finetune_forensic_transfer.py -m ft_c23_latent128_resnet3blocks_2classes_update_last -r 9 -i 5  &&

echo "Images:5 Run 10"
python finetune_forensic_transfer.py -m ft_c23_latent128_resnet3blocks_2classes_update_last -r 10 -i 5
#python calc_mean.py