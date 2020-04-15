import cv2
import os
import glob

img_array = []
dir_path_images = os.getcwd() + '/tsnes/8k_train_mean-3,3_c0images_val_images/*_both.png'
dir_path_videos = os.getcwd() + '/tsnes/8k_train_mean-3,3_c0images_val_images/video.avi'
filenames = [img for img in glob.glob(dir_path_images)]
a = sorted([filenames])
filenames.sort(key=int)
for filename in filenames:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(dir_path_videos, cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()