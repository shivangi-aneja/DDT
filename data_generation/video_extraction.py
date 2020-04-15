"""
Author: Andreas Rössler
"""
import os
import argparse
import cv2
import tqdm
from tqdm import tqdm
from os.path import join
import numpy as np
from imutils import face_utils
import json
import dlib
import glob
from os.path import isdir, isfile, join
from os import listdir


def get_non_zero_bb(img):
    # Get non zero elements for mask to get mask area
    a = np.where(img != 0)
    try:
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    except ValueError:
        return None
    return bbox


def get_iou(bb1, bb2):
    bb1 = {'x1': bb1[0], 'x2': bb1[1], 'y1': bb1[2], 'y2': bb1[3]}
    bb2 = {'x1': bb2[0], 'x2': bb2[1], 'y1': bb2[2], 'y2': bb2[3]}
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def write_out_images(data_path, mask_path, method_mask_path, output_path,
                     dataloader_output_path, label='original', **kwargs):
    comp = 'c23'     #['raw', 'c23', 'c40']

    for x in ['val']:
        print(x)
        with open('splits/{}.json'.format(x), 'r') as f:
            video_pairs = json.load(f)
        video_filenames = []
        for pair in video_pairs:
            video_filenames.append('_'.join(pair))
            video_filenames.append('_'.join(pair[::-1]))

        video_num = 0
        for video_fn in tqdm(video_filenames):
            if '595' in video_fn:
                print(video_fn)
                video_num += 1

                if label == 'original':
                    video_out_fn = video_fn.split('_')[0]
                    video_fn = video_out_fn + '.mp4'
                    mask_fn = video_out_fn + '.mp4'
                else:
                    video_out_fn = video_fn
                    video_fn = video_fn + '.mp4'
                    mask_fn = video_out_fn + '.mp4'

                # Video readers
                # video
                video_reader = cv2.VideoCapture(join(data_path, comp, 'videos', video_fn))
                # face tracking mask
                mask_cap_filenames = [img for img in glob.glob(mask_path + '/' + mask_fn.split('.')[0]+'/*.png')]
                mask_cap_filenames.sort()
                # method mask
                if method_mask_path is not None:
                    method_mask_cap = cv2.VideoCapture(join(method_mask_path, mask_fn))

                # Maximal number of frames
                num_frames = min(int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT)),int(len(mask_cap_filenames)))

                bounding_boxes = []
                frame_count = -1
                miss_detection = False
                masks = []

                for mask_cap_file in mask_cap_filenames:
                    frame_count += 1
                    mask_image = cv2.imread(mask_cap_file)
                    bb = get_non_zero_bb(mask_image)
                    if method_mask_path is not None:
                        method_mask_image = method_mask_cap.read()[1]
                        method_bb = get_non_zero_bb(method_mask_image)
                        if method_bb is not None:
                            iou = get_iou(bb, method_bb)
                            if iou < 0.1:
                                bb = method_bb
                                if not miss_detection:
                                    tqdm.write(video_fn)
                                miss_detection = True

                    if bb is not None:
                        bounding_boxes.append(bb)
                    masks.append(mask_image)

                    if frame_count >= num_frames - 1:
                        break
                if method_mask_path is not None:
                    method_mask_cap.release()

                # Write out mask images
                # mask_output_path = join(output_path, 'mask', x, label, video_out_fn)
                # os.makedirs(mask_output_path, exist_ok=True)

                # Output folders
                video_output_path = join(output_path, comp, x, label,  'images', video_out_fn)
                for apath in [video_output_path]:
                    os.makedirs(apath, exist_ok=True)

                cap = video_reader

                # Loop
                frame_count = -1
                while cap.isOpened():
                    frame_count += 1
                    if bounding_boxes[frame_count] is not None:
                        image = cap.read()[1]
                        bbox_frame = bounding_boxes[frame_count]
                        # For global
                        image_crop = image[bbox_frame[0]-15:bbox_frame[1]+15, bbox_frame[2]-15:bbox_frame[3]+15]
                        mask_image = masks[frame_count][bbox_frame[0]-15:bbox_frame[1]+15, bbox_frame[2]-15:bbox_frame[3]+15]
                        cv2.imwrite(join(video_output_path, '{:04d}.png'.format(frame_count)), image_crop)
                        # cv2.imwrite(join(mask_output_path, '{:04d}.png'.format(frame_count)), mask_image)

                    if frame_count >= len(bounding_boxes)-1:
                        break


def  write_out_images_dfdc(data_path, mask_path, method_mask_path, output_path,
                     dataloader_output_path, **kwargs):
    comp = 'c23'
    x = 'dfdc'
    video_num = 0
    video_filenames = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    for i, video_fn in enumerate(tqdm(video_filenames)):
        print(i)
        print(video_fn)
        video_num += 1
        dir_name = video_fn.split('.')[0]
        mask_fn = video_fn

        # Video readers
        video_reader = cv2.VideoCapture(join(data_path, video_fn))
        # method mask
        method_mask_reader = cv2.VideoCapture(join(method_mask_path, mask_fn))

        # Maximal number of frames
        # assert int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT)) == int(method_mask_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        # num_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        success_img, df_image = video_reader.read()
        success_mask, df_mask = method_mask_reader.read()
        count = 0
        # Write out mask images
        mask_output_path = join(output_path, 'mask', x, dir_name)
        os.makedirs(mask_output_path, exist_ok=True)

        # Output folders
        video_output_path = join(output_path, comp, x, 'images', dir_name)
        for apath in [video_output_path]:
            os.makedirs(apath, exist_ok=True)
        while success_img and success_mask:

            bbox_frame = get_non_zero_bb(df_mask)
            if bbox_frame is not None:
                image_crop = df_image[bbox_frame[0] - 30:bbox_frame[1] + 30, bbox_frame[2] - 30:bbox_frame[3] + 30]
                mask_image = df_mask[bbox_frame[0] - 30:bbox_frame[1] + 30,
                             bbox_frame[2] - 30:bbox_frame[3] + 30]
                cv2.imwrite(join(video_output_path, '{:04d}.png'.format(count)), image_crop)
                # cv2.imwrite(join(mask_output_path, '{:04d}.png'.format(count)), mask_image)
                count += 1

            success_img, df_image = video_reader.read()
            success_mask, df_mask = method_mask_reader.read()


def rect_to_bb(rect):
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()

    return x1, x2, y1, y2


def bb_to_oldbb(bb):
    return bb[2], bb[3], bb[0], bb[1]


def bb_to_center(bb):
    return bb[0] + int(0.5 * (bb[1]-bb[0])), bb[2] + int(0.5*(bb[3]-bb[2]))


def write_out_images_custom(data_path, output_path,
                      label='original', **kwargs):
    comp = 'c23'
    model_path = '/home/shivangi/Desktop/Projects/pretrained_models/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    for x in ['val']:
        print(x)
        with open('splits/{}.json'.format(x), 'r') as f:
            video_pairs = json.load(f)
        video_filenames = []
        for pair in video_pairs:
            video_filenames.append('_'.join(pair))
            video_filenames.append('_'.join(pair[::-1]))

        video_num = 0
        for video_fn in tqdm(video_filenames):
            if '115_939' in video_fn:
                print(video_fn)
                video_num += 1
                if label == 'original':
                    video_out_fn = video_fn.split('_')[0]
                    video_fn = video_out_fn + '.mp4'
                else:
                    video_out_fn = video_fn
                    video_fn = video_fn + '.mp4'

                # Video readers
                # video
                video_reader = cv2.VideoCapture(join(data_path, comp, 'videos', video_fn))

                # Maximal number of frames
                num_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))


                # Output folders
                video_output_path = join(output_path, comp, x, label, 'images', video_out_fn)
                for apath in [video_output_path]:
                    os.makedirs(apath, exist_ok=True)

                for frame_count in range(num_frames):
                    frame_image = video_reader.read()[1]
                    rects = detector(frame_image, 1)
                    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
                    image_crop = frame_image[y:y + h + 30, x:x + w + 20]
                    cv2.imwrite(join(video_output_path, '{:04d}.png'.format(frame_count)), image_crop)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', '-i',
                   default='/mnt/FaceForensics/v3/')
    p.add_argument('--output_path', '-vo', type=str,
                   default='/media/newhd/XYZ/FaceForensics/')
    p.add_argument('--dataloader_output_path', '-do', type=str,
                   default='/media/newhd/FaceForensics_video_crops_dataloader')
    p.add_argument('--name', '-n', type=str, default='DeepFakeDetection')
    p.add_argument('--mask_path', type=str, default='/mnt/FaceForensics/masks/')
    args = p.parse_args()

    if args.name == 'original':
        args.data_path = join(args.data_path, 'original_sequences')
        args.method_mask_path = None
        args.label = 'original'
    else:
        args.method_mask_path = join(args.data_path, 'manipulated_sequences', args.name, 'masks', 'videos')
        args.data_path = join(args.data_path, 'manipulated_sequences', args.name)
        args.label = join('manipulated', args.name)
    # if args.name == 'NeuralTextures':
    #    args.mask_path = join(args.mask_path, 'Face2Face', 'masks')
    # else:
    #    args.mask_path = join(args.mask_path, args.name, 'masks')
    # args.method_mask_path = '/media/newhd/DFDC/masks/'
    args.data_path = '/media/newhd/actors/'
    args.method_mask_path = '/media/newhd/actor_masks/'
    write_out_images_dfdc(**vars(args))
    # write_out_images(**vars(args))
