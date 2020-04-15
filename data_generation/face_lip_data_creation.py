import numpy as np
import torch
import cv2
import os
from os import makedirs
from os.path import isdir, join
from os import listdir
from torchvision import transforms
from common.utils.common_utils import get_all_frames, get_frame_list

DATASETS = {"ff"}


# if __name__ == "__main__":
#     manipulation = 'fs'
#     mode = 'val'
#     base_path_lip = '/media/newhd/data_mt/lip_crops/c23/' + mode + '/' + manipulation + '/'
#     base_path_face = '/media/newhd/data_mt/face_crops/c23/' + mode + '/' + manipulation + '/'
#     output_dir = '/home/shivangi/Desktop/Projects/master_thesis/data/combined_sequences_grayscale_new/c23/' + mode + '/' + manipulation + '/'
#     dir_list = [os.path.join(base_path_lip, f) for f in listdir(base_path_lip) if isdir(join(base_path_lip, f))]
#     print(manipulation)
#     for _dir in dir_list:
#         all_frames_list = get_all_frames(_dir)
#         first_frame_list = get_frame_list(dir_path=_dir, frame_cnt_per_video=30)
#         my_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
#         depth = 5
#         img_rows = 250
#         img_cols = 250
#
#         # Assign label to class
#         try:
#             lip_seq = torch.zeros([depth, img_rows, img_cols], dtype=torch.float64)
#
#             for i, img_path in enumerate(first_frame_list):
#                 if not os.path.isdir(output_dir + _dir.split('/')[-1] + '/'):
#                     makedirs(output_dir + _dir.split('/')[-1] + '/')
#                 req_idx = all_frames_list.index(img_path)
#                 for j in range(depth):
#                     frame_idx = req_idx + j
#                     input_img = cv2.imread(all_frames_list[frame_idx])
#                     input_img = np.array(input_img, dtype='uint8')
#                     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
#                     input_img = cv2.resize(input_img, (250, 250))
#                     input_img = my_transform(input_img)
#                     #lip_seq[3*j:3*j+3, :, :] = input_img
#                     lip_seq[j, :, :] = input_img
#
#                 face_img_path = img_path.replace('lip_crops', 'face_crops')
#                 face_img = cv2.imread(face_img_path)
#                 face_img = np.array(face_img, dtype='uint8')
#                 face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#                 torch.save((face_img, lip_seq), output_dir + _dir.split('/')[-1] + '/' + '{:04d}.pt'.format(i))
#         except:
#             print(all_frames_list[frame_idx])


if __name__ == "__main__":
    manipulation = 'nt'
    mode = 'val'
    base_path = '/media/newhd/google_df_podium_crops/c23/dfdc/images/'
    output_dir = '/media/newhd/google_df_podium_crops/c23/dfdc/image_crops/'
    dir_list = [os.path.join(base_path, f) for f in listdir(base_path) if isdir(join(base_path, f))]
    print(manipulation)

    for _dir in dir_list:
        first_frame_list = get_frame_list(dir_path=_dir, frame_cnt_per_video=30)

        for i, img_path in enumerate(first_frame_list):
            if not os.path.isdir(output_dir + _dir.split('/')[-1] + '/'):
                makedirs(output_dir + _dir.split('/')[-1] + '/')
            input_img = cv2.imread(img_path)
            cv2.imwrite(output_dir + _dir.split('/')[-1] + '/' + img_path.split('/')[-1], input_img)