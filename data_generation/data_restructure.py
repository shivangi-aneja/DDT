import json
import os
from os import makedirs

def restructure_dirs():
    base_path = '/media/newhd/FaceForensics_video_crops/c23/train/Face2Face/'
    with open('splits/train.json', 'r') as f:
        video_pairs = json.load(f)
        for pair in video_pairs:
            dir1 = pair[0]
            dir2 = pair[1]
            dir1_path = base_path + dir1
            dir2_path = base_path + dir2
            # Make directory path
            makedirs(dir1_path)
            makedirs(dir2_path)
            print(dir1, dir2)

            # Move files to specific directory
            bashCommand1 = 'mv ' + base_path + dir1 + '_* ' + base_path + dir1 + '/'
            os.system(bashCommand1)

            bashCommand2 = 'mv ' + base_path + dir2 + '_* ' + base_path + dir2 + '/'
            os.system(bashCommand2)


def create_split_file_list():
    manipulation = 'nt'
    dir_path = os.path.join('/media/newhd/FaceForensics_video_crops/c23/val/' + manipulation + '/')
    dir_list = list([x[0].split('/')[-1] for x in os.walk(dir_path)][1:])
    dir_list.sort(key=int)
    print(manipulation)

    f = open("splits/val_video.txt", "a+")
    for dir in dir_list:
        f.write("%s" % str(dir))
        f.write("\n")
    f.close()


if __name__ == "__main__":
    create_split_file_list()