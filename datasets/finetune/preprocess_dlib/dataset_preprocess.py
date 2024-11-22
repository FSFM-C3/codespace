import os
import sys
import argparse
import json
from config import cfg
import shutil
from tqdm import tqdm
import pandas as pd
import cv2

from tools.util import extract_face_from_fixed_num_frames
from tools.util import extract_and_save_face
align = False


def get_FF_video_split(split_json_file):
    with open(split_json_file, "r", encoding="utf-8") as f:
        content = json.load(f)
        video_split = [index for sublist in content for index in sublist]
    return video_split


def run_FF_real(compression='c23', num_frames=128):
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


def run_FF_fake(compression='c23', num_frames=32):
    video_dataset_split_index = {'train_videos_index': get_FF_video_split(cfg.FF_train_split),
                                 'val_videos_index': get_FF_video_split(cfg.FF_val_split),
                                 'test_videos_index': get_FF_video_split(cfg.FF_test_split)}

    for manipulation in cfg.FF_manipulation_list:
        src_dir = os.path.join(cfg.FF_fake_path, manipulation, compression, 'videos')
        if num_frames is None:
            dst_dir = os.path.join(cfg.FF_split_face_ds, 'all_frames',
                               cfg.FF_fake_path.split('FaceForensics/')[-1], manipulation, compression)
        else:
            dst_dir = os.path.join(cfg.FF_split_face_ds, str(num_frames) + '_frames',
                               cfg.FF_fake_path.split('FaceForensics/')[-1], manipulation, compression)
        print("splitting FF++ fake dataset, cropping and aligning face to", dst_dir)

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


def run_DFD(compression='raw', num_frames=32):
    for path in [cfg.DFD_real, cfg.DFD_fake]:
        src_dir = os.path.join(path, compression)
        dst_dir = os.path.join(cfg.DFD_split_face_ds, str(num_frames) + '_frames',
                               path.split('FaceForensics/')[-1], compression)
        dst_path = dst_dir
        os.makedirs(dst_path, exist_ok=True)
        print("splitting DFD dataset, cropping and aligning face to", dst_dir)

        for subdir, dirs, files in os.walk(src_dir):
            for video in tqdm(files):
                if video[-4:] == '.mp4':
                    src_video = os.path.join(subdir, video)
                    extract_face_from_fixed_num_frames(src_video, dst_path, video.split('.mp4')[0], num_frames, align=align)


def run_CelebDFv1(num_frames=32):
    sub_path = ['Celeb-real',
                'YouTube-real',
                'Celeb-synthesis']
    labels = [0, 0, 1]

    f = open(cfg.CelebDFv1_path + 'List_of_testing_videos.txt', 'r')  # official label for testing set
    test_content = f.readlines()
    test_videos = []
    for name in test_content:
        test_videos.append(name[2:].replace('\n', ''))  # remove '\n' in the official label

    for z in range(len(sub_path)):
        print("splitting CelebDFv1 dataset, cropping and aligning face to", cfg.CelebDFv1_split_face_ds)
        print('process in ', sub_path[z])
        video_type = 'real' if labels[z] == 0 else 'fake'

        # prepare test videos firstly:
        test_video_num = 0
        for subdir, dirs, files in os.walk(os.path.join(cfg.CelebDFv1_path, sub_path[z])):
            for file in tqdm(files):
                video = sub_path[z] + '/' + file
                if file[-4:] == '.mp4' and video in test_videos:
                    test_video_num += 1
                    dst_path = os.path.join(cfg.CelebDFv1_split_face_ds, str(num_frames)+'_frames', 'test', video_type)
                    os.makedirs(dst_path, exist_ok=True)
                    src_video = os.path.join(subdir, file)
                    video_name = sub_path[z] + '_' + file.split('.mp4')[0]
                    extract_face_from_fixed_num_frames(src_video, dst_path, video_name, num_frames, align=align)
        print("processed test videos:", test_video_num)

        # then split train/val videos:
        # i = 0
        # for subdir, dirs, files in os.walk(os.path.join(cfg.CelebDFv1_path, sub_path[z])):
        #     for file in tqdm(files):
        #         video = sub_path[z] + '/' + file
        #         if file[-4:] == '.mp4' and (video not in test_videos) and i < 0.8 * (len(files) - test_video_num):
        #             i += 1
        #             dst_path = os.path.join(cfg.CelebDFv1_split_face_ds, str(num_frames)+'_frames', 'train', video_type)
        #             os.makedirs(dst_path, exist_ok=True)
        #             src_video = os.path.join(subdir, file)
        #             video_name = sub_path[z] + '_' + file.split('.mp4')[0]
        #             extract_face_from_fixed_num_frames(src_video, dst_path, video_name, num_frames, align=align)
        #
        #         elif file[-4:] == '.mp4' and (video not in test_videos) and i < (len(files) - test_video_num):
        #             i += 1
        #             dst_path = os.path.join(cfg.CelebDFv1_split_face_ds, str(num_frames)+'_frames', 'val', video_type)
        #             os.makedirs(dst_path, exist_ok=True)
        #             src_video = os.path.join(subdir, file)
        #             video_name = sub_path[z] + '_' + file.split('.mp4')[0]
        #             extract_face_from_fixed_num_frames(src_video, dst_path, video_name, num_frames, align=align)


def run_CelebDFv2(num_frames=32):
    sub_path = ['Celeb-real',
                'YouTube-real',
                'Celeb-synthesis']
    labels = [0, 0, 1]

    f = open(cfg.CelebDFv2_path + 'List_of_testing_videos.txt', 'r')  # official label for testing set
    test_content = f.readlines()
    test_videos = []
    for name in test_content:
        test_videos.append(name[2:].replace('\n', ''))  # remove '\n' in the official label

    for z in range(len(sub_path)):
        print("splitting CelebDFv2 dataset, cropping and aligning face to", cfg.CelebDFv2_split_face_ds_nyr)
        print('process in ', sub_path[z])
        video_type = 'real' if labels[z] == 0 else 'fake'

        # prepare test videos firstly:
        test_video_num = 0
        for subdir, dirs, files in os.walk(os.path.join(cfg.CelebDFv2_path, sub_path[z])):
            for file in tqdm(files):
                video = sub_path[z] + '/' + file
                if file[-4:] == '.mp4' and video in test_videos:
                    test_video_num += 1
                    dst_path = os.path.join(cfg.CelebDFv2_split_face_ds_nyr, str(num_frames)+'_frames', 'test', video_type)
                    os.makedirs(dst_path, exist_ok=True)
                    src_video = os.path.join(subdir, file)
                    video_name = sub_path[z] + '_' + file.split('.mp4')[0]
                    extract_face_from_fixed_num_frames(src_video, dst_path, video_name, num_frames, align=align)
        print("processed test videos:", test_video_num)

        # # then split train/val videos:
        # i = 0
        # for subdir, dirs, files in os.walk(os.path.join(cfg.CelebDFv2_path, sub_path[z])):
        #     for file in tqdm(files):
        #         video = sub_path[z] + '/' + file
        #         if file[-4:] == '.mp4' and (video not in test_videos) and i < 0.8 * (len(files) - test_video_num):
        #             i += 1
        #             dst_path = os.path.join(cfg.CelebDFv2_split_face_ds, str(num_frames)+'_frames', 'train', video_type)
        #             os.makedirs(dst_path, exist_ok=True)
        #             src_video = os.path.join(subdir, file)
        #             video_name = sub_path[z] + '_' + file.split('.mp4')[0]
        #             extract_face_from_fixed_num_frames(src_video, dst_path, video_name, num_frames, align=align)
        #
        #         elif file[-4:] == '.mp4' and (video not in test_videos) and i < (len(files) - test_video_num):
        #             i += 1
        #             dst_path = os.path.join(cfg.CelebDFv2_split_face_ds, str(num_frames)+'_frames', 'val', video_type)
        #             os.makedirs(dst_path, exist_ok=True)
        #             src_video = os.path.join(subdir, file)
        #             video_name = sub_path[z] + '_' + file.split('.mp4')[0]
        #             extract_face_from_fixed_num_frames(src_video, dst_path, video_name, num_frames, align=align)


def run_DFDC(num_frames=32):
    print("splitting DFDC test dataset, cropping and aligning face to", cfg.DFDC_split_face_ds)
    label_df = pd.read_csv(cfg.DFDC_path + 'labels.csv', header=0)  # official label for testing set
    cls = ['real', 'fake']
    for subdir, dirs, files in os.walk(cfg.DFDC_path + 'ori_videos/'):  # 5000 videos from 5 folders in the test set
        for file in tqdm(files):
            if file[-4:] == '.mp4':
                label = (label_df[label_df['filename'] == file])['label']
                dst_path = os.path.join(cfg.DFDC_split_face_ds, str(num_frames)+'_frames', 'test', cls[int(label)])
                os.makedirs(dst_path, exist_ok=True)
                src_video = os.path.join(subdir, file)
                video_name = file.split('.mp4')[0]
                extract_face_from_fixed_num_frames(src_video, dst_path, video_name, num_frames, align=align)


def run_DFDC_P(num_frames=32):
    print("splitting DFDC test dataset, cropping and aligning face to", cfg.DFDC_P_split_face_ds)
    with open(os.path.join(cfg.DFDC_P_path, 'dataset.json'), 'r') as f:
        label_file = json.load(f)

    cls = ['real', 'fake']
    for c in range(len(cls)):
        dst_dir = os.path.join(cfg.DFDC_P_split_face_ds, str(num_frames)+'_frames', 'test', cls[c])
        os.makedirs(dst_dir, exist_ok=True)

    for video in tqdm(label_file.keys()):
        if label_file[video]["set"] == "test":
            label = label_file[video]["label"]
            src_video = os.path.join(cfg.DFDC_P_path, video)
            dst_path = os.path.join(cfg.DFDC_P_split_face_ds, str(num_frames)+'_frames', label_file[video]["set"], label)
            video_name = (video.split('/')[0] + '_' + video.split('/')[-1]).split('.mp4')[0]
            extract_face_from_fixed_num_frames(src_video, dst_path, video_name, num_frames, align=align)


def run_WildDeepfake():
    """it provides face images that have been cropped and aligned, just use it"""
    import shutil
    sub_path = ['fake_test',
                'real_test']
    for cls in range(len(sub_path)):
        dst_path = os.path.join(cfg.DFIW_split_face_ds, sub_path[cls].split('_')[1], sub_path[cls].split('_')[0])
        os.makedirs(dst_path, exist_ok=True)
        for subdir, dirs, files in os.walk(cfg.DFIW_path + sub_path[cls]):
            for file in tqdm(files):
                if file[-4:] == '.png':
                    shutil.copyfile(os.path.join(subdir, file),
                                    os.path.join(dst_path,
                                                 subdir.replace('/', '_').split('deepfake_in_the_wild_')[1] +
                                                 '_frame_' + file))


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


def make_DiFF_set_extract_face(split_set):
    # src:
    fake_set_path = '../' + split_set
    fake_sub_path = ['FS', 'FE', 'I2I', 'T2I']
    real_set_path = '../DiFF_real/' + split_set
    # dst:
    split_ds_path = "../DiFF_" + split_set + "_faces/"

    for cls in range(len(fake_sub_path)):

        dst_path = os.path.join(split_ds_path, fake_sub_path[cls], split_set)
        dst_path_real = os.path.join(dst_path, 'DiFF_real')
        dst_path_fake = os.path.join(dst_path, 'fake')
        os.makedirs(dst_path_real, exist_ok=True)
        os.makedirs(dst_path_fake, exist_ok=True)

        for subdir, dirs, files in tqdm(os.walk(os.path.join(fake_set_path, fake_sub_path[cls]))):
            for file in files:
                if file[-4:] == '.jpg' or '.png':
                    img = cv2.imread(os.path.join(subdir, file))
                    if img is not None:
                        save_name = os.path.join(subdir, file).split(fake_sub_path[cls])[1].replace('/', '_')
                        save_name = save_name.replace('.', '_').replace('_png', '.png')
                        # unify to .png image file
                        # save_name = save_name.replace('.jpg', '.png')
                        save_name = save_name[:-4] + '.png'
                        # shutil.copy(src=os.path.join(subdir, file), dst=os.path.join(dst_path_fake, save_name))
                        extract_and_save_face(img, dst_path_fake, save_name)

        for subdir, dirs, files in tqdm(os.walk(real_set_path)):
            for file in files:
                if file[-4:] == '.jpg' or '.png':
                    img = cv2.imread(os.path.join(subdir, file))
                    if img is not None:
                        # shutil.copy(src=os.path.join(real_test_path, file), dst=os.path.join(dst_path_real, file))
                        extract_and_save_face(img, dst_path_real, file)


def get_args_parser():
    parser = argparse.ArgumentParser('FSFM_C3 data preprocessing', add_help=False)
    parser.add_argument('--dataset', default='FF++',
                        help='choose from [FF++, DFD, CelebDFv1, CelebDFv2, DFDC, DFDCP, WildDeepfake, DiFF]')
    parser.add_argument('--num_frames', default=None,
                        help='the number of frames extracted per video, extract 32 frames per video by default')

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.dataset == 'FF++':
        # FF_all_binary_cls_ds: 128 frames per real videos VS 32 per fake (4 types exclude Fsh) for data balance:
        run_FF_real(compression='c23', num_frames=args.num_frames * 4)
        run_FF_fake(compression='c23', num_frames=args.num_frames)
        src_path_real = cfg.FF_split_face_ds + '128_frames/original_sequences/youtube/'
        src_path_fake = cfg.FF_split_face_ds + '32_frames/manipulated_sequences/'
        dst_path = cfg.FF_split_face_ds + '32_frames/DS_FF++_all_cls/'
        gen_FF_all_binary_cls_ds(src_path_real, src_path_fake, dst_path, compression='c23')
    elif args.dataset == 'DFD':
        # DFD_binary_cls_ds: 32 frames pre real videos VS 32 frames pre real:
        run_DFD(compression='raw', num_frames=args.num_frames)
        src_path_real = cfg.DFD_split_face_ds + '32_frames/original_sequences/actors/'
        src_path_fake = cfg.DFD_split_face_ds + '32_frames/manipulated_sequences/DeepFakeDetection/'
        dst_path = cfg.DFD_split_face_ds + '32_frames/DS_DFD_binary_cls/'
        gen_DFD_binary_cls_ds(src_path_real, src_path_fake, dst_path, compression='raw')
    elif args.dataset == 'CelebDFv1':
        run_CelebDFv1(num_frames=args.num_frames)
    elif args.dataset == 'CelebDFv2':
        run_CelebDFv2(num_frames=args.num_frames)
    elif args.dataset == 'DFDC':
        run_DFDC(num_frames=args.num_frames)
    elif args.dataset == 'DFDCP':
        run_DFDC_P(num_frames=args.num_frames)
    elif args.dataset == 'WildDeepfake':
        run_WildDeepfake()
    elif args.dataset == 'DiFF':
        make_DiFF_set_extract_face(split_set='val')
        make_DiFF_set_extract_face(split_set='test')
    else:
        print('Invalid Input')        