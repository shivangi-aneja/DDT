#!/usr/bin/env python
""" Downloads FaceForensics++ public data release
Example usage:
    see -h or https://github.com/ondyari/FaceForensics
"""
# -*- coding: utf-8 -*-
import argparse
import os
import urllib
import urllib.request
import tempfile
import time
import sys
import json
import random
from tqdm import tqdm
from os.path import join

# URLs
SERVER_URL = 'http://kaldir.vc.in.tum.de/FaceForensics/'
TOS_URL = SERVER_URL + 'webpage/FaceForensics_TOS.pdf'
BASE_URL = SERVER_URL + 'v2/'
FILELIST_URL = BASE_URL + 'misc/filelist.json'
VIDEOLENGTHS_URL = BASE_URL + 'misc/video_lengths.json'
DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_A.h5.bk',
                         'decoder_B.h5', 'decoder_B.h5.bk',
                         'encoder.h5', 'encoder.h5.bk']

# Types
DATASETS = {
    'original_youtube_videos': BASE_URL + 'misc/downloaded_youtube_videos.zip',
    'original': 'original_sequences',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'Neuraltextures': 'manipulated_sequences/Neuraltextures',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection'}
ALL_DATASETS = {'original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'DeepFakeDetection'}
COMPRESSION = ['raw', 'c0', 'c23', 'c40']
TYPE = ['images', 'videos', 'masks', 'models']


def download_files(filenames, base_url, output_path, report_progress=True):
    os.makedirs(output_path, exist_ok=True)
    if report_progress:
        filenames = tqdm(filenames)
    for filename in filenames:
        download_file(base_url + filename, join(output_path, filename))


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\rProgress: %d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download_file(url, out_file, report_progress=False):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        if report_progress:
            urllib.request.urlretrieve(url, out_file_tmp,
                                       reporthook=reporthook)
        else:
            urllib.request.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        tqdm.write('WARNING: skipping download of existing file ' + out_file)


def get_folder_length(folder, video_lengths, dataset):
    # Video length is dependent on dataset
    if dataset == 'original':  # Simple look up
        folder_length = video_lengths[folder]
    else:
        target, source = folder.split('_')
        if dataset == 'Deepfakes':
            # Deepfakes manipulates all images from target
            folder_length = video_lengths[target]
        elif dataset == 'Face2Face':
            # Face2Face manipulates as many images as the source
            # provides
            folder_length = video_lengths[source]
        elif dataset == 'FaceSwap':
            # FaceSwap simply transfers expressions and thus uses
            # the minimum
            folder_length = min(
                video_lengths[source],
                video_lengths[target]
            )
        else:
            raise Exception
    return folder_length


def main():
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics v2 public data release.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-o','--output_path', type=str, help='Output directory.')
    parser.add_argument('-d', '--dataset', type=str, default='all',
                        help='Which dataset to download, either pristine or '
                             'manipulated data or the downloaded youtube '
                             'videos.',
                        choices=list(DATASETS.keys()) + ['all']
                        )
    parser.add_argument('-c', '--compression', type=str, default='c0',
                        help='Which compression degree. All compressed videos '
                             'have been generated with h264 with a varying '
                             'codec.',
                        choices=COMPRESSION
                        )
    parser.add_argument('-t', '--type', type=str, default='videos',
                        help='Which file type, i.e. videos, images, masks or, '
                             'for Deepfakes, models.',
                        choices=TYPE
                        )
    parser.add_argument('--num_images', type=int, default=None,
                        help='Only applied when downloading images. When '
                             'specified, the script downloads only <num_images>'
                             ' images per video, reducing the download time '
                             'immensely.'
                        )
    parser.add_argument('--seed', type=int, default=None,
                        help='Only applied when downloading images and '
                             'if num_images is specified. Fixes a seed for the '
                             'random images selected by the script so that the '
                             'results are reproduceable. Useful e.g., if '
                             'one wants to downloads specific image and mask'
                             'pairs.'
                        )
    args = parser.parse_args()

    # TOS
    print('By pressing any key to continue you confirm that you have agreed '\
          'to the FaceForensics terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    _ = input('')

    # Seed
    if args.seed is not None:
        random.seed(args.seed)

    # Extract arguments
    c_datasets = [args.dataset] if args.dataset != 'all' else ALL_DATASETS
    c_type = args.type
    c_compression = args.compression
    output_path = args.output_path

    # Check for special dataset cases
    for dataset in c_datasets:
        dataset_path = DATASETS[dataset]
        # Special cases
        if 'youtube' in dataset_path:
            # Here we download the original youtube videos zip file
            print('Downloading original youtube videos.')
            print('Please be patient, this may take a while (~38.5gb)')
            download_file(dataset_path,
                          out_file=join(output_path,
                                        'downloaded_videos.zip'),
                          report_progress=True)
            return

        # Else: regular datasets
        print('Downloading {} of dataset "{}" in quality {}...'.format(
            c_type, dataset_path, c_compression
        ))

        # Get filelists and video lenghts list from server
        video_lengths = json.loads(urllib.request.urlopen(VIDEOLENGTHS_URL).read().decode("utf-8"))
        if 'original' in dataset_path:
            filelist = ['{:03d}'.format(i) for i in range(1000)]
        else:
            # Load filelist from server
            file_pairs = json.loads(urllib.request.urlopen(FILELIST_URL).read().decode("utf-8"))
            # Get filelist
            filelist = []
            for pair in file_pairs:
                filelist.append('_'.join(pair))
                if c_type != 'models':
                    filelist.append('_'.join(pair[::-1]))

        # Server and local paths
        dataset_base_url = BASE_URL + '{}/{}/{}/'.format(
            dataset_path, c_compression, c_type)
        dataset_output_path = join(output_path, dataset_path, c_compression,
                                   c_type)
        print('Output path: {}'.format(dataset_output_path))

        if c_type == 'videos':
            if c_compression == 'raw':
                print('Raw data is only available in image format. Aborting.')
                return
            filelist = [filename + '.mp4' for filename in filelist]
            download_files(filelist, dataset_base_url, dataset_output_path)
        # Else: images and manipulation method masks
        else:
            # Exceptions
            if dataset == 'original':
                if c_type not in ['videos', 'images']:
                    if args.dataset != 'all':
                        print('Only images and videos available for '
                              'original data. Aborting.')
                        return
                    else:
                        print('Only images and videos available for '
                              'original data. Skipping original.\n')
                        continue
            if dataset != 'Deepfakes' and c_type == 'models':
                print('Models only available for Deepfakes. Aborting')
                return
            if c_compression != 'raw' and c_type in ['masks', 'models']:
                print('Masks and models are only available for raw images.')
                print('Continuing with compression type raw.')
                dataset_base_url = dataset_base_url.replace(c_compression,
                                                            'raw')
                dataset_output_path = dataset_output_path.replace(c_compression,
                                                                  'raw')

            # Get video lengths from server to retrieve filenames
            for folder in tqdm(filelist):
                if c_type != 'models':
                    folder_length = get_folder_length(folder, video_lengths,
                                                      dataset)
                    folder_filelist = ['{:04d}.png'.format(i)
                                       for i in range(folder_length)]
                    # Maybe limit the number of downloaded images for each video
                    if args.num_images is not None:
                        assert args.num_images > 0
                        folder_filelist = random.sample(folder_filelist,
                                                        args.num_images)
                else:
                    folder_filelist = DEEPFAKES_MODEL_NAMES

                # Folder paths
                folder_base_url = dataset_base_url + folder + '/'
                folder_dataset_output_path = join(dataset_output_path,
                                                  folder)
                download_files(folder_filelist, folder_base_url,
                               folder_dataset_output_path,
                               report_progress=False)   # already done


if __name__ == "__main__":
    main()