
import cv2
import os
from os import listdir
from os.path import isfile, join

filepath = '/media/newhd/videos_c40/original/'
file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))]
dir_name = "/media/newhd/cropped_frames_orig_c40"

for file in file_list:

    #################### Setting up the file ################
    videoFile = cv2.VideoCapture(filepath + file)
    # Find the number of frames
    video_length = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    #success, image = videoFile.read()

    try:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    except OSError:
        print('Error: Creating directory of data')

    frameId=0
    while videoFile.isOpened():
        # Extract the frame
        ret, frame = videoFile.read()
        # Write the results back to output location.
        name = dir_name + '/' + str(file.split('.')[0]) + '_' + str(int(frameId)) + '.png'
        print('Creating : ' + name)
        cv2.imwrite(name, frame)
        count = count + 1
        frameId += 1
        # If there are no more frames left
        if count > (video_length - 1):
            # Release the feed
            videoFile.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            break

    #################### Setting up parameters ################
    # OpenCV is notorious for not being able to good to
    # predict how many frames are in a video. The point here is just to
    # populate the "desired_frames" list for all the individual frames
    # you'd like to capture.
    # fps = videoFile.get(cv2.CAP_PROP_FPS)
    # est_video_length_minutes = 0.10  # Round up if not sure.
    # est_tot_frames = 30     #int(est_video_length_minutes * 60 * fps)  # Sets an upper bound # of frames in video clip
    #
    # n = 1  # Desired interval of frames to include
    # desired_frames = n * np.arange(est_tot_frames)
    #
    # #################### Initiate Process ################
    #
    # for i in desired_frames:
    #     videoFile.set(1, i - 1)
    #     success, image = videoFile.read(1)  # image is an array of array of [R,G,B] values
    #     frameId = videoFile.get(1)  # The 0th frame is often a throw-away
    #     name = os.getcwd() + '/data/' + dir_name + '/' + str(file.split('.')[0]) + '_' + str(int(frameId)) + '.png'
    #     print('Creating : ' + name)
    #     cv2.imwrite(name, image)

    # When everything done, release the capture
    videoFile.release()
    cv2.destroyAllWindows()
    print("Done file : " + str(file))
