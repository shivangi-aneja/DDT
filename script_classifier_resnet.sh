#!/bin/bash 

# activate the virtual environment
source ../env/venv/bin/activate

echo "Images:100 Run 1"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last  -r 1 -i 100 &&

echo "Images:100 Run 2"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last  -r 2 -i 100 &&

echo "Images:100 Run 3"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 3 -i 100  &&

echo "Images:100 Run 4"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 4 -i 100  &&

echo "Images:100 Run 5"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 5 -i 100  &&

echo "Images:100 Run 6"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 6 -i 100  &&

echo "Images:100 Run 7"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 7 -i 100  &&

echo "Images:100 Run 8"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 8 -i 100  &&

echo "Images:100 Run 9"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 9 -i 100  &&

echo "Images:100 Run 10"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 10 -i 100
#python calc_mean.py