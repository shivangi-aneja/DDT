#!/bin/bash 

# activate the virtual environment
source ../env/venv/bin/activate

echo "Images:25 Run 1"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last  -r 1  &&

echo "Images:25 Run 2"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last  -r 2  &&

echo "Images:25 Run 3"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 3   &&

echo "Images:25 Run 4"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 4   &&

echo "Images:25 Run 5"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 5   &&

echo "Images:25 Run 6"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 6   &&

echo "Images:25 Run 7"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 7   &&

echo "Images:25 Run 8"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 8   &&

echo "Images:25 Run 9"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 9   &&

echo "Images:25 Run 10"
python finetune_classifier.py -m classifier_c23_latent128_resnet3blocks_2classes_update_last -r 10