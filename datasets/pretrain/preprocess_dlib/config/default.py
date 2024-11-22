# -*- coding: utf-8 -*-

from yacs.config import CfgNode as CN

_C = CN()

# =====================Face param=====================
_C.face_scale = 1.3  # follow FF++'s bounding box size multiplier to get a bigger face region
_C.face_size = 224
_C.img_format = '.png'

# =====================FF++=====================
# ***ori label path
_C.FF_train_split = '../FaceForensics/dataset/splits/train.json'
_C.FF_val_split = '../FaceForensics/dataset/splits/val.json'
_C.FF_test_split = '../FaceForensics/dataset/splits/test.json'
# ***ori data path
_C.FF_real_path = '../FaceForensics/original_sequences/youtube/'
_C.FF_fake_path = '../FaceForensics/manipulated_sequences/'
_C.FF_manipulation_list = ['DeepFakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
# ***specific path to save split ds of face
_C.FF_split_face_ds = '../FaceForensics/face_dataset_split/dlib_no_align_no_resize/'

# ===================YoutubeFace=========================
# ***ori data path
_C.YTFace_path = '../YoutubeFace/frame_images_DB/'
# ***specific path to save extracted face
_C.YTFace_face_ds = '../YoutubeFace/faces/'

# ===================VGGFace2=========================
# ***ori data path
_C.VGGFace2_path = ['../VGGFace2/train', '../VGGFace2/test']
# ***specific path to save extracted face
_C.VGGFace2_face_ds = '../VGGFace2/faces/'

# ==========Face parsing(for pretrain)=================
# FF++ real faces:
_C.FF_real_face_paths_for_parsing = \
    ['../FaceForensics/face_dataset_split/dlib_no_align_no_resize/128_frames/original_sequences/youtube/c23/train/',
     '../FaceForensics/face_dataset_split/dlib_no_align_no_resize/128_frames/original_sequences/youtube/c23/val/']
# save results to this path, which for Pretrain:
_C.FF_face_parse_ds_path = \
    '../pretrain_datasets/FaceForensics_youtube/all_frames/c23/'

# YoutubeFace:
_C.YTFace_path_for_parsing = '../YoutubeFace/faces/'
# save results to this path, which for Pretrain:
_C.YTFace_parse_ds_path = '../pretrain_datasets/YoutubeFace/'

# VGGFace2 faces:
_C.VGGFace2_path_for_parsing = '../VGGFace2/faces/'
# save results to this path, which for Pretrain:
_C.VGGFace2_parse_ds_path = '../pretrain_datasets/VGGFace2/'

_C.face_img_dir = 'images'  # the subdir of face_parse_ds_path for save ori img
_C.pm_format = '.npy'  # format of parsing map from face, with the same name as ori img
_C.face_pm_dir = 'parsing_maps'  # the subdir of face_parse_ds_path for saving parse maps
_C.vis_pm_dir = 'vis_parsing_maps'  # the subdir of face_parse_ds_path for saving parse maps vis