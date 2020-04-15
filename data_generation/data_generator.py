#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for creating .npy files for image
"""

from __future__ import division
import os
import numpy as np
from PIL import Image
import cv2


# Resize png images to be of size 512*512
def resize_images(input_path):
    new_size = (256, 256)
    # For images
    for image in sorted(os.listdir(input_path)):
        if image.endswith(".png"):
            img = cv2.imread(input_path + image)
            new_image = cv2.resize(img, new_size)
            new_image = cv2.cvtColor(new_image, cv2.IMREAD_COLOR)
            cv2.waitKey(0)
            cv2.imwrite(input_path + image.split(".")[0] + '.png', new_image)


def save_to_numpy_array(input_path, output_path, mode):
    image_list = []

    # For images
    for image in sorted(os.listdir(input_path)):

        if image.endswith(".png"):
            image = Image.open(input_path + image)
            # This data has shape (height, width, channels)
            data = np.array(image, dtype='uint8')
            # Change to (channels, height, width)
            data = np.transpose(data, [2, 0, 1])
            image_list.append(data)

    name = mode + '_data.npy'
    np.save(os.path.join(output_path + name), image_list)


def main(train_dir, val_dir, dataset):
    input_dir_train = train_dir
    input_dir_val = val_dir
    output_dir_train = os.path.join(os.getcwd(), 'data/ff/train_c23/f2f/')
    output_dir_val = os.path.join(os.getcwd(), 'data/ff/val_c23/f2f/')

    # resize images
    resize_images(input_path=output_dir_train)
    resize_images(input_path=output_dir_val)

    # Saving data to numpy file
    # save_to_numpy_array(input_path=input_dir_train_f2f, output_path=output_dir_f2f, mode='face2face')
    # save_to_numpy_array(input_path=input_dir_train_orig, output_path=output_dir_orig, mode='original')


def combine_npy_files():
    train_f2f_X = os.getcwd() + '/data_latent/ff/train_c0_f2f_X.npy'
    train_orig_X = os.getcwd() + '/data_latent/ff/train_c0_orig_X.npy'
    train_X = os.getcwd() + '/data_latent/ff/train_c0_X.npy'
    with open(train_X, "wb") as f_handle:
        data_train_x = np.concatenate((np.load(train_f2f_X), np.load(train_orig_X)))
        np.save(f_handle, data_train_x)

    train_f2f_Y = os.getcwd() + '/data_latent/ff/train_c0_f2f_Y.npy'
    train_orig_Y = os.getcwd() + '/data_latent/ff/train_c0_orig_Y.npy'
    train_Y = os.getcwd() + '/data_latent/ff/train_c0_Y.npy'
    with open(train_Y, "wb") as f_handle:
        data_train_y = np.concatenate((np.load(train_f2f_Y), np.load(train_orig_Y)))
        np.save(f_handle, data_train_y)

    val_f2f_X = os.getcwd() + '/data_latent/ff/val_c0_f2f_X.npy'
    val_orig_X = os.getcwd() + '/data_latent/ff/val_c0_orig_X.npy'
    val_X = os.getcwd() + '/data_latent/ff/val_c0_X.npy'
    with open(val_X, "wb") as f_handle:
        data_val_x = np.concatenate((np.load(val_f2f_X), np.load(val_orig_X)))
        np.save(f_handle, data_val_x)

    val_f2f_Y = os.getcwd() + '/data_latent/ff/val_c0_f2f_Y.npy'
    val_orig_Y = os.getcwd() + '/data_latent/ff/val_c0_orig_Y.npy'
    val_Y = os.getcwd() + '/data_latent/ff/val_c0_Y.npy'
    with open(val_Y, "wb") as f_handle:
        data_val_y = np.concatenate((np.load(val_f2f_Y), np.load(val_orig_Y)))
        np.save(f_handle, data_val_y)


if __name__ == '__main__':
    # combine_npy_files()
    # train_dir = 'train_c40'
    # val_dir = 'val_c40'
    # dataset = 'orig'
    # main(train_dir, val_dir, dataset)
    resize_images(input_path='/home/shivangi/Desktop/Projects/master_thesis/data/ff_all/c23/val_all_c23/nt/')
