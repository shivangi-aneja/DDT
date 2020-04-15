import os
import cv2
from os import makedirs
import numpy as np

manipulation = 'nt'
input_dir = os.path.join('/media/newhd/FaceForensics_video_crops/c23/train/' + manipulation + '/')
output_dir = os.path.join('/media/newhd/optical_flow/c23/train/' + manipulation + '/')
dir_list = [x[0] for x in os.walk(input_dir)][1:]
print(manipulation)
my_val = False
for dir in dir_list:
    dir = dir.split('/')[-1]
    if not os.path.isdir(output_dir + dir + '/'):
        makedirs(output_dir + dir + '/')

    if dir == '105':
        my_val = True
        continue
    if my_val:
        print(dir)
        frames = [image for image in sorted(os.listdir(input_dir + dir + '/'))]
        frame1 = cv2.imread(input_dir + dir+'/' + frames[0])
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

        for frame in frames[1:]:
            # print(frame)
            frame2 = cv2.imread(input_dir + dir+'/' + frame)
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(output_dir + dir+'/' + frame, rgb)
            prvs = next
        print("Done")
