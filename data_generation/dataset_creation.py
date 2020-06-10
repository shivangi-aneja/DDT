import cv2
from os import makedirs
import dlib
from imutils import face_utils
from os.path import isdir, join, isfile
from os import listdir
from tqdm import tqdm
import os
from common.utils.common_utils import get_frame_list, rect_to_bb


model_path = '/home/shivangi/Desktop/Projects/pretrained_models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)


def write_out_images_dfdc(video_path, image_path):

    video_filenames = [f for f in listdir(video_path) if isfile(join(video_path, f))]
    for i, video_fn in enumerate(tqdm(video_filenames)):

        dir_name = video_fn.split('.')[0]
        # Video readersdir_name
        video_reader = cv2.VideoCapture(join(video_path, video_fn))
        # num_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read the first frame
        success_img, image_crop = video_reader.read()
        count = 0
        # Output folders
        video_output_path = join(image_path, dir_name)

        for apath in [video_output_path]:
            os.makedirs(apath, exist_ok=True)

        while success_img:
            cv2.imwrite(join(video_output_path, '{:04d}.png'.format(count)), image_crop)
            count += 1
            success_img, image_crop = video_reader.read()


def split_images(source_path, dest_path):
    dir_list = [os.path.join(source_path, f) for f in listdir(source_path) if isdir(join(source_path, f))]
    for _dir in dir_list:
        print(_dir)
        first_frame_list = get_frame_list(dir_path=_dir, frame_cnt_per_video=30)

        for i, img_path in enumerate(first_frame_list):
            if not os.path.isdir(dest_path + _dir.split('/')[-1] + '/'):
                makedirs(dest_path + _dir.split('/')[-1] + '/')
            input_img = cv2.imread(img_path)
            cv2.imwrite(dest_path + _dir.split('/')[-1] + '/' + img_path.split('/')[-1], input_img)


def create_face_crops(src_path, dest_path):
    dir_list = [os.path.join(src_path, f) for f in listdir(src_path) if isdir(join(src_path, f))]
    for dir in dir_list:
        dir_image_path = [os.path.join(f) for f in listdir(dir) if isdir(join(dir))]
        print(dir)
        if '0' in dir:
            for image in sorted(dir_image_path):
                if image.endswith(".png"):
                    img = cv2.imread(dir + '/' + image)
                    rects = detector(img, 1)
                    # rects, _ = mtcnn.detect(img)
                    if len(rects) == 0:
                        print("Failed :",  dir + '/' + image)
                    else:
                        for i in range(len(rects)):
                            (x, y, w, h) = face_utils.rect_to_bb(rects[i])
                            y_min = max(0, y - 30)
                            x_min = max(0, x - 30)
                            face_img = img[y_min:y + h + 30, x_min:x + w + 30]
                            face_img = cv2.resize(face_img, (256, 256))
                            os.makedirs(dest_path + dir.split('/')[-1], exist_ok=True)
                            cv2.imwrite(dest_path + dir.split('/')[-1] + '/' + image.split(".")[0] + '_' + str(i) + '.png', face_img)


if __name__ == "__main__":
    # json_file_path = '/media/newhd/Facebook/deepfake-detection-challenge/train.json'
    video_path = '/media/newhd/FaceDatasets/Dessa/c23/videos/train/fake/'

    image_path = '/media/newhd/XYZ/FaceForensics/c23/dfdc/fake/'
    image_path_30 = '/media/newhd/XYZ/FaceForensics/c23/dfdc/fake_30/'

    src_path = '/media/newhd/FaceDatasets/FaceForensics/c23/face_crops/c23/test/Deepfakes/'
    dest_path = '/media/newhd/FaceImages/Dessa/c23/face_crops/train/fake/'
    # write_out_videos_dfdc(json_file_path=json_file_path, video_path=video_path, output_path=output_path)
    # write_out_images_dfdc(video_path, image_path)
    split_images(source_path=image_path, dest_path=image_path_30)
    # create_face_crops(src_path, dest_path)
