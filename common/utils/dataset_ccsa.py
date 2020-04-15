import random
import os
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import PIL.ImageOps


# Initialization.Create_Pairs
class CustomDataSet(Dataset):
    def __init__(self, base_path, num_classes, src_fake_classes, image_count, tgt_img_list, transform=None, should_invert=False):

        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()
        self.tgt_img_list = tgt_img_list

        real_path = base_path + 'orig/'
        df_path = base_path + 'df/'
        fs_path = base_path + 'fs/'
        f2f_path = base_path + 'f2f/'
        nt_path = base_path + 'nt/'

        real_images = [os.path.join(real_path, f) for f in listdir(real_path) if isfile(join(real_path, f))]
        df_images = [os.path.join(df_path, f) for f in listdir(df_path) if isfile(join(df_path, f))]
        fs_images = [os.path.join(fs_path, f) for f in listdir(fs_path) if isfile(join(fs_path, f))]
        f2f_images = [os.path.join(f2f_path, f) for f in listdir(f2f_path) if isfile(join(f2f_path, f))]
        nt_images = [os.path.join(nt_path, f) for f in listdir(nt_path) if isfile(join(nt_path, f))]

        fake_image_list_map = {'df': df_images, 'fs': fs_images, 'f2f': f2f_images, 'nt': nt_images}

        self.src_image_list = None

        if num_classes == 2:
            real_list = list(real_images)
            fake_list = list(fake_image_list_map[src_fake_classes[0]])
            if image_count != 'all':
                real_list = random.sample(real_list, image_count)
                fake_list = random.sample(fake_list, image_count)
            self.src_image_list = list(real_list) + list(fake_list)

        elif num_classes == 3:
            real_list = list(real_images)
            fake1_list = list(fake_image_list_map[src_fake_classes[0]])
            fake2_list = list(fake_image_list_map[src_fake_classes[1]])
            if image_count != 'all':
                real_list = random.sample(real_list, image_count)
                fake1_list = random.sample(fake1_list, image_count)
                fake2_list = random.sample(fake2_list, image_count)
            self.src_image_list = list(real_list) + list(fake1_list) + list(fake2_list)

        elif num_classes == 4:
            real_list = list(real_images)
            fake1_list = list(fake_image_list_map[src_fake_classes[0]])
            fake2_list = list(fake_image_list_map[src_fake_classes[1]])
            fake3_list = list(fake_image_list_map[src_fake_classes[2]])
            if image_count != 'all':
                real_list = random.sample(real_list, image_count)
                fake1_list = random.sample(fake1_list, image_count)
                fake2_list = random.sample(fake2_list, image_count)
                fake3_list = random.sample(fake3_list, image_count)
            self.src_image_list = list(real_list) + list(fake1_list) + list(fake2_list) + list(fake3_list)

        elif num_classes == 5:
            real_list = list(real_images)
            fake1_list = list(fake_image_list_map[src_fake_classes[0]])
            fake2_list = list(fake_image_list_map[src_fake_classes[1]])
            fake3_list = list(fake_image_list_map[src_fake_classes[2]])
            fake4_list = list(fake_image_list_map[src_fake_classes[3]])
            if image_count != 'all':
                real_list = random.sample(real_list, image_count)
                fake1_list = random.sample(fake1_list, image_count)
                fake2_list = random.sample(fake2_list, image_count)
                fake3_list = random.sample(fake3_list, image_count)
                fake4_list = random.sample(fake4_list, image_count)
            self.src_image_list = list(real_list) + list(fake1_list) + list(fake2_list) + list(fake3_list) + list(
                fake4_list)

        training_pos = []
        training_neg = []
        for trs in range(len(self.src_image_list)):
            for trt in range(len(self.tgt_img_list)):
                src_class = 0 if "orig" in self.src_image_list[trs] else 1
                tgt_class = 0 if "orig" in self.tgt_img_list[trt] else 1
                if src_class == tgt_class:
                    training_pos.append([trs, trt, 1])
                else:
                    training_neg.append([trs, trt, 0])

        self.imgs = training_pos + training_neg
        # random.shuffle(training_neg)
        # self.imgs = Training_P + Training_N[:3 * len(Training_P)]
        random.shuffle(self.imgs)

    def __getitem__(self, idx):

        src_idx, tgt_idx, domain = self.imgs[idx]
        x_src, y_src, x_tgt, y_tgt = None, None, None, None
        try:
            x_src = cv2.imread(self.src_image_list[src_idx])
            x_src = np.array(x_src, dtype='uint8')
            x_src = cv2.cvtColor(x_src, cv2.COLOR_BGR2RGB)
            y_src = 0 if "orig" in self.src_image_list[src_idx] else 1

            x_tgt = cv2.imread(self.tgt_img_list[tgt_idx])
            x_tgt = np.array(x_tgt, dtype='uint8')
            x_tgt = cv2.cvtColor(x_tgt, cv2.COLOR_BGR2RGB)
            y_tgt = 0 if "orig" in self.tgt_img_list[tgt_idx] else 1

            if self.should_invert:
                x_src = PIL.ImageOps.invert(x_src)
                x_tgt = PIL.ImageOps.invert(x_tgt)

            if self.transform is not None:
                x_src = self.transform(x_src)
                x_tgt = self.transform(x_tgt)
            else:
                x_src = self.to_tensor(x_src)
                x_tgt = self.to_tensor(x_tgt)

        except:
            print(self.src_image_list[src_idx])
            print(self.tgt_img_list[tgt_idx])
            exit()

        return x_src, y_src, x_tgt, y_tgt

    def __len__(self):
        return len(self.imgs)