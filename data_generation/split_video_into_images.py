'''
Using OpenCV takes a videos and produces a number of images.

Requirements
----
You require OpenCV 3.2 to be installed.

Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py

Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, join

filepath = os.getcwd()+"/data/videos_c23/f2f_c23/"
file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))]
dir_name = "cropped_frames_f2f_c23"

for file in file_list:
    try:
        if not os.path.exists(os.getcwd() + '/data/' + dir_name):
            os.makedirs(os.getcwd() + '/data/' + dir_name)
    except OSError:
        print('Error: Creating directory of data')

    #################### Setting up the file ################
    videoFile = cv2.VideoCapture(filepath + file)
    success, image = videoFile.read()
    count = 0

    while success:
        name = os.getcwd() + '/data/' + dir_name + '/' + str(file.split('.')[0]) + str(int(count)) + '.png'
        cv2.imwrite(name, image)  # save frame as JPEG file
        success, image = videoFile.read()
        print('Read a new frame: ', success)
        count += 1

    # When everything done, release the capture
    videoFile.release()
    cv2.destroyAllWindows()
    print("Done file : " + str(file))
