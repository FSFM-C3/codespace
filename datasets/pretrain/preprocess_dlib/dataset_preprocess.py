import os
import sys
import argparse
import json
from config import cfg
import shutil
from tqdm import tqdm
import pandas as pd
import cv2
import multiprocessing

from tools.util import extract_face_from_fixed_num_frames
from tools.util import extract_and_save_face
align = False


def get_FF_video_split(split_json_file):
    with open(split_json_file, "r", encoding="utf-8") as f:
        content = json.load(f)
        video_split = [index for sublist in content for index in sublist]
    return video_split


def run_FF_real(compression='c23', num_frames=32):
    video_dataset_split_index = {'train_videos_index': get_FF_video_split(cfg.FF_train_split),
                                 'val_videos_index': get_FF_video_split(cfg.FF_val_split),
                                 'test_videos_index': get_FF_video_split(cfg.FF_test_split)}

    src_dir = os.path.join(cfg.FF_real_path, compression, 'videos')
    if num_frames is None:
        dst_dir = os.path.join(cfg.FF_split_face_ds, 'all_frames',
                               cfg.FF_real_path.split('FaceForensics/')[-1], compression)
    else:
        dst_dir = os.path.join(cfg.FF_split_face_ds, str(num_frames) + '_frames',
                           cfg.FF_real_path.split('FaceForensics/')[-1], compression)
    print("splitting FF++ real dataset, cropping and aligning face to", dst_dir)

    for subdir, dirs, files in os.walk(src_dir):
        for video in tqdm(files):
            if video[-4:] == '.mp4':
                src_video = os.path.join(subdir, video)
                if video[:3] in video_dataset_split_index['train_videos_index']:
                    split = 'train'
                elif video[:3] in video_dataset_split_index['val_videos_index']:
                    split = 'val'
                elif video[:3] in video_dataset_split_index['test_videos_index']:
                    split = 'test'
                dst_path = os.path.join(dst_dir, split)
                os.makedirs(dst_path, exist_ok=True)
                extract_face_from_fixed_num_frames(src_video, dst_path, video.split('.mp4')[0], num_frames, align=align)


def mp_extract(subdir, files, src_path, dst_path):
    for file in files:
        img_path = os.path.join(subdir, file)
        img = cv2.imread(img_path)
        if img is not None:
            save_name = os.path.relpath(img_path, os.path.dirname(src_path)).replace('/', '_')
            extract_and_save_face(img, dst_path, save_name)


def extract_face_VGGFace2(src_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    tasks = []

    for subdir, dirs, files in os.walk(src_path):
        if files:
            tasks.append(pool.apply_async(mp_extract, args=(subdir, files, src_path, dst_path)))

    for task in tqdm(tasks):
        task.wait()

    pool.close()
    pool.join()


def run_VGGFace2():
    for p in cfg.VGGFace2_path:
        extract_face_VGGFace2(p, cfg.VGGFace2_face_ds)


def get_args_parser():
    parser = argparse.ArgumentParser('FSFM_C3 data preprocessing', add_help=False)
    parser.add_argument('--dataset', default='FF++',
                        help="choose 'FF++' or 'VGGFace2'")
    parser.add_argument('--compression', default='c23',
                        help="only for FF++, compression rate for videos, choose 'c23' (by default) or 'c40'")
    parser.add_argument('--num_frames', default=None,
                        help='only for FF++, the number of frames extracted per video, extract all frames by default')

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.dataset == 'FF++':
        run_FF_real(compression=args.compression, num_frames=args.num_frames)
    else:
        run_VGGFace2()