# -*- coding: utf-8 -*-

import os
import shutil
from tqdm import tqdm
from config import cfg


def cp_img(src_path, dst_path, file_prefix):
    os.makedirs(dst_path, exist_ok=True)
    for subdir, dirs, files in os.walk(src_path):
        for file in tqdm(files):
            if file[-4:] == cfg.img_format:
                src = os.path.join(subdir, file)
                dst = os.path.join(dst_path, file_prefix + '_' + file)
                shutil.copyfile(src, dst)


def gen_FF_all_binary_cls_ds(src_real, src_fake, dst_path, compression):
    for subset in ['train', 'val', 'test']:
        src_real_path = os.path.join(src_real, compression, subset)
        dst_real_path = os.path.join(dst_path, compression, subset, 'real_youtube')
        cp_img(src_real_path, dst_real_path, file_prefix='youtube')

        for manipulation in cfg.FF_manipulation_list:
            if manipulation == 'FaceShifter':  # we do not use Fsh for training/finetune
                continue
            src_fake_path = os.path.join(src_fake, manipulation, compression, subset)
            dst_fake_path = os.path.join(dst_path, compression, subset, 'fake_four')
            cp_img(src_fake_path, dst_fake_path, file_prefix=manipulation)


def gen_FF_each_binary_cls_ds(src_real, src_fake, dst_path, compression):
    for subset in ['train', 'val', 'test']:
        src_real_path = os.path.join(src_real, compression, subset)
        for manipulation in cfg.FF_manipulation_list:
            dst_real_path = os.path.join(dst_path, compression, manipulation, subset, 'real_youtube')
            src_fake_path = os.path.join(src_fake, manipulation, compression, subset)
            dst_fake_path = os.path.join(dst_path, compression, manipulation, subset, 'fake_' + manipulation)
            cp_img(src_real_path, dst_real_path, file_prefix='youtube')
            cp_img(src_fake_path, dst_fake_path, file_prefix=manipulation)


def gen_FSh_binary_cls_ds(src_real, src_fake, dst_path, compression):
    for subset in ['train', 'val', 'test']:
        src_real_path = os.path.join(src_real, compression, subset)
        manipulation = 'FaceShifter'
        dst_real_path = os.path.join(dst_path, compression, manipulation, subset, 'real_youtube')
        src_fake_path = os.path.join(src_fake, manipulation, compression, subset)
        dst_fake_path = os.path.join(dst_path, compression, manipulation, subset, 'fake_' + manipulation)
        cp_img(src_real_path, dst_real_path, file_prefix='youtube')
        cp_img(src_fake_path, dst_fake_path, file_prefix=manipulation)


def gen_DFD_binary_cls_ds(src_real, src_fake, dst_path, compression):
    """
    use all videos in DFD as test set
    """
    src_real_path = os.path.join(src_real, compression)
    src_fake_path = os.path.join(src_fake, compression)
    dst_real_path = os.path.join(dst_path, compression, 'test', 'real_actors')
    dst_fake_path = os.path.join(dst_path, compression, 'test', 'fake_DeepFakeDetection')
    cp_img(src_real_path, dst_real_path, file_prefix='actors')
    cp_img(src_fake_path, dst_fake_path, file_prefix='DeepFakeDetection')


if __name__ == '__main__':
    # # FF_all_binary_cls_ds: 128 frames per real videos VS 32 per fake (4 types exclude Fsh) for data balance:
    # src_path_real = cfg.FF_split_face_ds + '128_frames/original_sequences/youtube/'
    # src_path_fake = cfg.FF_split_face_ds + '32_frames/manipulated_sequences/'
    # dst_path = cfg.FF_split_face_ds + '32_frames/DS_FF++_all_cls/'
    # gen_FF_all_inary_cls_ds(src_path_real, src_path_fake, dst_path, compression='c40')

    # # FF_all_binary_cls_ds: 32 frames per real videos VS 32 frames per fake:
    # src_path_real = cfg.FF_split_face_ds + '32_frames/original_sequences/youtube/'
    # src_path_fake = cfg.FF_split_face_ds + '32_frames/manipulated_sequences/'
    # dst_path = cfg.FF_split_face_ds + '32_frames/DS_FF++_each_cls/'
    # gen_FF_each_binary_cls_ds(src_path_real, src_path_fake, dst_path, compression='c23')

    # FSh_binary_cls_ds: 32 frames pre real videos VS 32 frames pre real:
    src_path_real = cfg.FF_split_face_ds + '32_frames/original_sequences/youtube/'
    src_path_fake = cfg.FF_split_face_ds + '32_frames/manipulated_sequences/'
    dst_path = cfg.FF_split_face_ds + '32_frames/DS_FF++_each_cls/'
    gen_FSh_binary_cls_ds(src_path_real, src_path_fake, dst_path, compression='c23')

    # # DFD_binary_cls_ds: 32 frames pre real videos VS 32 frames pre real:
    # src_path_real = cfg.DFD_split_face_ds + '32_frames/original_sequences/actors/'
    # src_path_fake = cfg.DFD_split_face_ds + '32_frames/manipulated_sequences/DeepFakeDetection/'
    # dst_path = cfg.DFD_split_face_ds + '32_frames/DS_DFD_binary_cls/'
    # gen_DFD_binary_cls_ds(src_path_real, src_path_fake, dst_path, compression='raw')