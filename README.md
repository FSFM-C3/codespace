# FSFM-C3
This is the official code space for FSFM-C3. In this repository, we expose all the source code of the project and provide a detailed deployment explanation. 
% Please visit [this website](https://huggingface.co/spaces/FSFM-C3/Masking) for the **mask strategies demo** proposed in the paper.
# Environment
```bash
conda create -n fsfmc3 python=3.9
conda activate fsfmc3
pip install -r requirements.txt
```
# Model Pre-training
We pre-trained the model based on **FaceForensics++ (FF++)** / **VGGFace2 (VF2)** datasets. We will introduce the implementation of model pre-training in detail. 
## Data Pre-processing
### Dataset Preparation
The datasets used in the model pre-training process are public resources and should be downloaded in advance. The location of the datasets in the file directory is described in ***File Structure***.
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [VGGFace2](https://github.com/ox-vgg/vgg_face2)
### Toolkit Preparation
We use **DLIB** for face detection and cropping with a 30% addition size from each side, as well as the **FACER** toolkit for face parsing instead of alignment. Both toolkits are available on the public web. The location of the toolkits in the file directory is described in ***File Structure***.
- [DLIB](https://github.com/codeniko/shape_predictor_81_face_landmarks)
- [FACER](https://github.com/FacePerceiver/facer)
### File Structure
The following is the **recommended file structure**. The files in each directory are described in the comments. You can also choose to customize the file structure by changing the corresponding directory information in `/datasets/pretrain/preprocess_dlib/config/default.py` (be careful not to change the parameter name).

> `*_real*`：Raw real data path
> 
> `*_fake*`：Raw fake data path
> 
> `*_split_face_ds`：Data after frame extraction
> 
> `*_for_parsing`：Pretraining data before face parsing
> 
> `*_parse_ds_path`：Pretraining data after face parsing (ready for model pretraining)

```bash
datasets/pretrain/
├── FaceForensics/	# FF++ 
│   ├── dataset/	# FF++ data partition
│   │   └── splits/
│   │       ├── train.json
│   │       ├── val.json
│   │       └── test.json
│   ├── original_sequences/	# FF++ real data
│   │   └── youtube/
│   ├── manipulated_sequences/	# FF++ fake data
│   │   ├── DeepFakes/
│   │   ├── Face2Face/
│   │   ├── FaceSwap/
│   │   └── NeuralTextures/
│   └── face_dataset_split/	# data after frame extraction (automatic creating)
│
├── VGGFace2/
│   ├── train/	# raw data for training
│   ├── test/	# raw data for testing
│   └── faces/	# data after face extraction (train + test) (automatic creating)
│
├── pretrain_datasets/	# data for pretraining (automatic creating)
│   ├── FaceForensics_youtube/	# FF++ data for pretraining
│   └── VGGFace2/	# VGGFace2 data for pretraining
│
└── preprocess_dlib/
    ├── config/
    │   ├── __init__.py
    │   └── default.py	# define file structure
    ├── tools/	# toolkits for face parsing
    │   ├── facer/	# FACER toolkit
    │   ├── shape_predictor_81_face_landmarks.dat	# DLIB toolkit
    │   └── util.py
    ├── dataset_preprocess.py	# for frame extraction
    └── face_parse.py	# for face parsing
```
### Frame Extraction
We use the full **VGGFace2 (VF2)** dataset for pre-training, alongside original YouTube videos (pristine faces) from **FaceForensics++ (FF++)**, i.e., FF++ original (FF++_o). The FF++_o includes 720 training and 140 validation videos. 
We recommend these different pre-training data scales:

- 10W：the c23 (HQ) version of FF++_o, includes 720 training and 140 validation videos, extract 128 frames per video
- 50W：the c23 (HQ) version of FF++_o, includes 720 training and 140 validation videos, extract all frames of the video
- 300W：VGGFace2, including the full training and testing subsets

Run the program `/datasets/pretrain/preprocess_dlib/dataset_preprocess.py` to automatically extract video frames:

```python
# 10W
python dataset_preprocess.py \
	--dataset FF++ \
	--compression 'c23'
	--num_frames 128
# 50W
python dataset_preprocess.py \
	--dataset FF++ \
	--compression 'c23'
# 300W
python dataset_preprocess.py \
	--dataset VGGFace2 
```

> **Note that for the FF++ dataset**, it involves extraction methods with different frame numbers. It is necessary to modify the corresponding directory information in `/datasets/pretrain/preprocess_dlib/config/default.py` according to the actual selection:

```python
# 10W
_C.FF_real_face_paths_for_parsing = \
	['../FaceForensics/face_dataset_split/dlib_no_align_no_resize/128_frames/original_sequences/youtube/c23/train/',
     '../FaceForensics/face_dataset_split/dlib_no_align_no_resize/128_frames/original_sequences/youtube/c23/val/']
_C.FF_face_parse_ds_path = \
    '../pretrain_datasets/FaceForensics_youtube/128_frames/c23/'
# 50W
_C.FF_real_face_paths_for_parsing = \
	['../FaceForensics/face_dataset_split/dlib_no_align_no_resize/all_frames/original_sequences/youtube/c23/train/',
     '../FaceForensics/face_dataset_split/dlib_no_align_no_resize/all_frames/original_sequences/youtube/c23/val/']
_C.FF_face_parse_ds_path = \
    '../pretrain_datasets/FaceForensics_youtube/all_frames/c23/'
```
### Frame Processing
We use **DLIB** for face detection and cropping with a 30% addition size from each side, as well as the **FACER** toolkit for face parsing instead of alignment. Cropped faces are resized to 224×224, and parsing maps are saved as binary stream files, enabling efficient CRFR-P masking during pre-training.

Run the program `/datasets/pretrain/preprocess_dlib/face_parse.py` to automatically performs frame processing:
 
```python
# FF++
python face_parse.py --dataset FF++
# VGGFace2
python face_parse.py --dataset VGGFace2 
```
## Training
Run the program `/FSFM_C3/main_pretrain.py` to automatically pre-trains the model:

```bash
# 10W
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=2 main_pretrain.py \
    --batch_size 256 \
    --accum_iter 8 \
    --epochs 400 \
    --model fsfm_vit_small_patch16 \
    --input_size 224 \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --weight_sfr 0.007 \
    --weight_cl 0.1 \
    --cl_loss SimSiam \
    --weight_decay 0.05 \
    --blr 1.5e-4 \
    --warmup_epochs 40 \
    --pretrain_data_path /datasets/pretrain_datasets/FaceForensics_youtube/128_frames/c23/

# 50W
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=2 main_pretrain.py \
    --batch_size 256 \
    --accum_iter 8 \
    --epochs 400 \
    --model fsfm_vit_base_patch16 \
    --input_size 224 \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --weight_sfr 0.007 \
    --weight_cl 0.1 \
    --cl_loss SimSiam \
    --weight_decay 0.05 \
    --blr 1.5e-4 \
    --warmup_epochs 40 \
    --pretrain_data_path /datasets/pretrain_datasets/FaceForensics_youtube/all_frames/c23/

# 300W
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=2 main_pretrain.py \
    --batch_size 256 \
    --accum_iter 8 \
    --num_workers 10 \
    --epochs 400 \
    --model fsfm_vit_base_patch16 \
    --input_size 224 \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --weight_sfr 0.007 \
    --weight_cl 0.1 \
    --cl_loss SimSiam \
    --weight_decay 0.05 \
    --blr 1.5e-4 \
    --warmup_epochs 40 \
    --pretrain_data_path /datasets/pretrain_datasets/VGGFace2/
```
# Downstream Tasks
The pre-trained model is able to be fine-tuned on different datasets to perform different downstream tasks. We will introduce the implementation of model fine-tuning in detail. 
## Data Pre-processing
### Dataset Preparation
The datasets used in the downstream tasks are public resources and should be downloaded in advance. The location of the datasets in the file directory is described in ***File Structure***.
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DeepFake (v1 & v2)](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DeepFake Detection Challenge (Full & Preview)](https://ai.meta.com/datasets/dfdc/)
- [deepfake in the wild](https://github.com/OpenTAI/wild-deepfake)
### File Structure
The following is the **recommended file structure**. The files in each directory are described in the comments. You can also choose to customize the file structure by changing the corresponding directory information in `/datasets/downstream/preprocess_dlib/config/default.py` (be careful not to change the parameter name).

```bash
datasets/downstream/
├── Celeb-DF/	# Celeb-DF (v1)
│   ├── Celeb-real/
│   ├── YouTube-real/
│   ├── Celeb-synthesis/
│   └── face_dataset_split/	# data after frame extraction (automatic creating)
│
├── Celeb-DF-v2/	# Celeb-DF (v2) 
│   ├── Celeb-real/
│   ├── YouTube-real/
│   ├── Celeb-synthesis/
│   └── face_dataset_split/	# data after frame extraction (automatic creating)
│
├── deepfake_in_the_wild/	# deepfake in the wild
│   ├── real_train/ 			
│   ├── real_test/ 
│   ├── fake_train/ 
│   ├── fake_test/  
│   └── face_dataset_split/	# data after frame extraction (automatic creating)
│
├── DFDC/# DeepFake Detection Challenge (Full)
│   ├── test/  
│   └── face_dataset_split/	# data after frame extraction (automatic creating)
│
├── DFDCP/	# DeepFake Detection Challenge (Preview)
│   ├── dataset.json	
│   ├── method_A/
│   ├── method_B/
│   ├── original_videos/
│   └── face_dataset_split/	# data after frame extraction (automatic creating)
│
├── FaceForensics/	# FF++ 
│   ├── dataset/	# FF++ data partition
│   │   └── splits/ 
│   │       ├── train.json
│   │       ├── val.json
│   │       └── test.json
│   ├── original_sequences/	# FF++ real data
│   │  	└── youtube/
│   ├── manipulated_sequences/	# FF++ fake data
│   │   ├── DeepFakes/
│   │   ├── Face2Face/
│   │   ├── FaceSwap/
│   │   └── NeuralTextures/
│   └──face_dataset_split/	# data after frame extraction (automatic creating)
│
└── preprocess_dlib/
    ├── config/
    │   ├── __init__.py
    │   └── default.py	# define file structure
    ├── tools/	# toolkits for face parsing
    │   ├── facer/	# FACER toolkit
    │   ├── shape_predictor_81_face_landmarks.dat	# DLIB toolkit
    │   └── util.py
    ├── dataset_preprocess.py	# for frame extraction
    └── face_parse.py	# for face parsing
```

### Frame Extraction
Run the program `/datasets/downstream/preprocess_dlib/dataset_preprocess.py` to automatically extract video frames:

```python
python dataset_preprocess.py --dataset [FF++, DFD, CelebDFv1, CelebDFv2, DFDC, DFDCP, WildDeepfake]
```
### Frame Processing
We use **DLIB** for face detection and cropping with a 30% addition size from each side, as well as the **FACER** toolkit for face parsing instead of alignment. Cropped faces are resized to 224×224, and parsing maps are saved as binary stream files, enabling efficient CRFR-P masking during pre-training.

Run the program `/datasets/downstream/preprocess_dlib/face_parse.py` to automatically performs frame processing:
 
```python
python face_parse.py --dataset [FF++, DFD, CelebDFv1, CelebDFv2, DFDC, DFDCP, WildDeepfake]
```
## Task 1: Deepfake Detection
### Fine-tuning
Run the program `/FSFM_C3/main_finetune.py` to automatically pre-trains the model:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=2 main_finetune.py \
    --accum_iter 1 \
    --apply_simple_augment \
    --batch_size 32 \
    --nb_classes 2 \
    --model vit_base_patch16 \
    --finetune [your pretained model ckpt] \
    --epochs 100 \
    --blr 2.5e-4 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval \
    --data_path /deepfake_datatsets/FaceForensics/face_dataset_split/dlib_no_align_no_resize/32_frames/DS_FF++_all_cls/c23
```

## Task 2: DiFF Detection
### Fine-tuning
Run the program `/FSFM_C3/main_finetune_DiFF.py` to automatically pre-trains the model:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=1 main_finetune_DiFF.py \
    --accum_iter 1 \
    --apply_simple_augment \
    --batch_size 256 \
    --nb_classes 2 \
    --model vit_base_patch16 \
    --finetune [your pretained model ckpt] \
    --epochs 10 \
    --blr 5e-4 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval \
    --data_path /deepfake_datatsets/FaceForensics/face_dataset_split/dlib_no_align_no_resize/32_frames/DS_FF++_all_cls/c40 \
```
