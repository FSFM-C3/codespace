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

# ================DFD(from FF++)===============
# ***ori data path
_C.DFD_real = '../FaceForensics/original_sequences/actors/'
_C.DFD_fake = '../FaceForensics/manipulated_sequences/DeepFakeDetection/'
# ***specific path to save split ds of face
_C.DFD_split_face_ds = '../FaceForensics/face_dataset_split/dlib_no_align_no_resize/'

# ===================CelebDFV1=========================
# ***ori data path
_C.CelebDFv1_path = '../Celeb-DF/'
# ***specific path to save split ds of face
_C.CelebDFv1_split_face_ds = '../Celeb-DF/face_dataset_split/dlib_no_align_no_resize/'
_C.CelebDFv1_split_face_ds_nyr = '../Celeb-DF/face_dataset_split/dlib_no_align_no_resize_nyr/'
# ===================CelebDFV2=========================
# ***ori data path
_C.CelebDFv2_path = '../Celeb-DF-v2/'
# ***specific path to save split ds of face
_C.CelebDFv2_split_face_ds = '../Celeb-DF-v2/face_dataset_split/dlib_no_align_no_resize/'
_C.CelebDFv2_split_face_ds_nyr = '../Celeb-DF-v2/face_dataset_split/dlib_no_align_no_resize_nyr/'

# ===================DFDC=========================
# ***ori data path
_C.DFDC_path = '../DFDC/test/'
# ***specific path to save split ds of face
_C.DFDC_split_face_ds = '../DFDC/face_dataset_split/dlib_no_align_no_resize/'

# ===================DFDC_Preview=========================
# ***ori data path
_C.DFDC_P_path = '../DFDCP/'
# ***specific path to save split ds of face
_C.DFDC_P_split_face_ds = '../DFDCP/face_dataset_split/dlib_no_align_no_resize/'

# ===================WildDeepfake/i.e., DFIW=========================
# ***ori data path
_C.DFIW_path = '../deepfake_in_the_wild/'
# ***specific path to save split ds of face
_C.DFIW_split_face_ds = '../deepfake_in_the_wild/face_dataset_split/'

_C.face_img_dir = 'images'  # the subdir of face_parse_ds_path for save ori img
_C.pm_format = '.npy'  # format of parsing map from face, with the same name as ori img
_C.face_pm_dir = 'parsing_maps'  # the subdir of face_parse_ds_path for saving parse maps
_C.vis_pm_dir = 'vis_parsing_maps'  # the subdir of face_parse_ds_path for saving parse maps vis