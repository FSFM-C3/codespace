# FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning
Welcome to the official GitHub repository for the **FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning**. We've shared the complete source code of our project. This repository also includes comprehensive deployment instructions to guide you through the reproduction process.
# Environment
Git clone our repository, creating a python environment and activate it via the following command: 
```bash
conda create -n fsfmc3 python=3.9
conda activate fsfmc3
pip install -r requirements.txt
```
# Model Pre-training
Our model has been pre-trained on different datasets including **FaceForensics++ (FF++)**, **YoutubeFace (YTF)**, and **VGGFace2 (VF2)**. In the following sections, we will detail how to implement model pre-training.
## Data Pre-processing
### Dataset Preparation
Download the following datasets that we utilized during the pre-training. The location of the datasets in the file directory is described in ***File Structure***.
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [YoutubeFace](https://www.cs.tau.ac.il/~wolf/ytfaces/)
- [VGGFace2](https://github.com/ox-vgg/vgg_face2)
### Toolkit Preparation
We use **DLIB** for face detection, as well as the **FACER** toolkit for face parsing. Download both toolkits in advance. The location of the toolkits in the file directory is described in ***File Structure***.
- [DLIB](https://github.com/codeniko/shape_predictor_81_face_landmarks)
- [FACER](https://github.com/FacePerceiver/facer)
### File Structure
The following is the **recommended file structure**. The files in each directory are described in the comments. You can customize the file structure by changing the corresponding directory information in `/datasets/pretrain/preprocess_dlib/config/default.py`.
```bash
datasets/
├── FaceForensics/    # FF++
│   ├── dataset/    # FF++ data partition
│   │   └── splits/
│   │       ├── train.json
│   │       ├── val.json
│   │       └── test.json
│   ├── original_sequences/    # FF++ real data
│   │   └── youtube/
│   ├── manipulated_sequences/    # FF++ fake data
│   │   ├── DeepFakes/
│   │   ├── Face2Face/
│   │   ├── FaceSwap/
│   │   └── NeuralTextures/
│   └── face_dataset_split/    # data after frame extraction (automatic creating)
│
├── YoutubeFace/    # YoutubeFace
│   ├── frame_images_DB/    # raw data from YoutubeFace
│   └── faces/    # data after face extraction (automatic creating)
│
├── VGGFace2/    # VGGFace2
│   ├── train/    # raw data for training
│   ├── test/    # raw data for testing
│   └── faces/    # data after face extraction (train + test) (automatic creating)
│
├── pretrain_datasets/    # data for pretraining (automatic creating)
│   ├── FaceForensics_youtube/    # FF++ data for pretraining
│   ├── YoutubeFace/    # YoutubeFace data for pretraining
│   └── VGGFace2/    # VGGFace2 data for pretraining
│
└── pretrain/preprocess_dlib/
    ├── config/
    │   ├── __init__.py
    │   └── default.py    # define file structure
    ├── tools/
    │   ├── facer/    # FACER toolkit
    │   ├── shape_predictor_81_face_landmarks.dat    # DLIB toolkit
    │   └── util.py
    ├── dataset_preprocess.py    # for frame extraction
    └── face_parse.py    # for face parsing
```
### Frame Extraction
We use the full **YoutubeFace (YTF)**/**VGGFace2 (VF2)** dataset for pre-training, alongside original YouTube videos (pristine faces) from **FaceForensics++ (FF++)**, i.e., FF++ original (FF++_o), for ablation experiment. 
- 10W：the c23 (HQ) version of FF++_o, includes 720 training and 140 validation videos, extract 128 frames per video
- 60W：YouTubeFace, including 3,425 videos from YouTube, broken to frames
- 300W：VGGFace2, including the full training and testing subsets

Run the program `/datasets/pretrain/preprocess_dlib/dataset_preprocess.py` to extract video frames/people faces:
```python
python dataset_preprocess.py --dataset [FF++, YTF, VF2]
```
### Frame Processing
We use **DLIB** for face detection and cropping with a 30% addition size from each side, as well as the **FACER** toolkit for face parsing instead of alignment. Cropped faces are resized to 224×224, and parsing maps are saved as binary stream files, enabling efficient CRFR-P masking during pre-training.

Run the program `/datasets/pretrain/preprocess_dlib/face_parse.py` to perform frame processing:
```python
python face_parse.py --dataset [FF++, YTF, VF2]
```
## Training
Run the program `/FSFM_C3/main_pretrain.py` to train the model:

```bash
# VGGFace2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=4 main_pretrain.py \
    --batch_size 256 \
    --accum_iter 4 \
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
    --pretrain_data_path ../pretrain_datasets/VGGFace2/  \
    --output_dir [path to save your model ckpt and logs]

# YouTubeFace
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=4 main_pretrain.py \
    --batch_size 256 \
    --accum_iter 4 \
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
    --pretrain_data_path ../pretrain_datasets/YoutubeFace/ \
    --output_dir [path to save your model ckpt and logs]
```
# Downstream Tasks
The pre-trained model is able to be fine-tuned on relevant datasets to perform different downstream tasks. In the following sections, we will introduce the implementation of model fine-tuning. 
## Task 1: Deepfake Detection
To evaluate the generalizability of our method across diverse DfD scenarios, we follow the challenging cross-dataset setup. Specifically, we fine-tune one detector on the **FaceForensics++ (FF++, c23/HQversion)** dataset and test it on unseen datasets: CelebDF-v2 (CDFv2), Deepfake Detection Challenge (DFDC), Deepfake Detection Challenge preview (DFDCp), and Wild Deepfake(WDF).
### Dataset Preparation
Download the following datasets that we utilized during the fine-tuning. The location of the datasets in the file directory is described in ***File Structure***.
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DeepFake (v1 & v2)](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DeepFake Detection Challenge (Full & Preview)](https://ai.meta.com/datasets/dfdc/)
- [deepfake in the wild](https://github.com/OpenTAI/wild-deepfake)
### File Structure
The following is the **recommended file structure**. The files in each directory are described in the comments. You can customize the file structure by changing the corresponding directory information in `/datasets/finetune/preprocess_dlib/config/default.py`.
```bash
datasets/
├── Celeb-DF/    # Celeb-DF (v1)
│   ├── Celeb-real/ 				
│   ├── YouTube-real/ 				
│   ├── Celeb-synthesis/ 			
│   └── face_dataset_split/    # data after frame extraction (automatic creating)
│
├── Celeb-DF-v2/    # Celeb-DF (v2) 
│   ├── Celeb-real/ 				
│   ├── YouTube-real/ 				
│   ├── Celeb-synthesis/ 			
│   └── face_dataset_split/    # data after frame extraction (automatic creating)
│
├── deepfake_in_the_wild/    # deepfake in the wild
│   ├── real_train/ 			
│   ├── real_test/ 
│   ├── fake_train/ 
│   ├── fake_test/  
│   └── face_dataset_split/    # data after frame extraction (automatic creating)
│
├── DFDC    # DeepFake Detection Challenge (Full)
│   ├── test/  
│   └── face_dataset_split/    # data after frame extraction (automatic creating)
│
├── DFDCP/    # DeepFake Detection Challenge (Preview)
│   ├── dataset.json	
│   ├── method_A/
│   ├── method_B/
│   ├── original_videos/
│   └── face_dataset_split/    # data after frame extraction (automatic creating)
│
├── FaceForensics/    # FF++
│   ├── dataset/    # FF++ data partition
│   │   └── splits/ 
│   │       ├── train.json
│   │       ├── val.json
│   │       └── test.json
│   ├── original_sequences/    # FF++ real data
│   │  	└── youtube/
│   ├── manipulated_sequences/    # FF++ fake data
│   │   ├── DeepFakes/
│   │   ├── Face2Face/
│   │   ├── FaceSwap/
│   │   └── NeuralTextures/
│   └──face_dataset_split/    # data after frame extraction (automatic creating)
│
└── finetune/preprocess_dlib/
    ├── config/
    │   ├── __init__.py
    │   └── default.py    # define file structure
    ├── tools/
    │   └── util.py
    └── dataset_preprocess.py    # for frame extraction
```
### Frame Extraction
Run the program `/datasets/finetune/preprocess_dlib/dataset_preprocess.py` to extract video frames:
```python
python dataset_preprocess.py --dataset [FF++, DFD, CelebDFv1, CelebDFv2, DFDC, DFDCP, WildDeepfake]
```
### Fine-tuning
Run the program `/FSFM_C3/main_finetune.py` to fine-tune the model:
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=2 main_finetune.py \
    --accum_iter 1 \
    --apply_simple_augment \
    --batch_size 32 \
    --nb_classes 2 \
    --model vit_base_patch16 \
    --finetune [your pretained model ckpt] \
    --epochs 10 \
    --blr 2.5e-4 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval \
    --data_path /datasets/FaceForensics/face_dataset_split/dlib_no_align_no_resize/32_frames/DS_FF++_all_cls/c23 \
    --output_dir [path to save your model ckpt and logs]
```
### Evaluation
Run the program `/FSFM_C3/main_test_DF.py` to evaluate the model:
```python
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=20 main_test_DF.py \
    --eval \
    --resume [your fine-tuned model ckpt] \
    --model vit_base_patch16 \
    --nb_classes 2 \
    --batch_size 320
```
## Task 2: DiFF Detection
To further investigate the adaptability of our method against emerging unknown facial forgeries, we adopt cross-distribution testing using the recently released **DiFF** benchmark. This dataset contains high-quality face images synthesized by SOTA diffusion models across four sub-sets: T2I (Text-to-Image), I2I (Image-to-Image), FS (Face Swapping), and FE (Face Editing). We train only on the **FF++_DeepFake (c23) subset**.
### Dataset Preparation
Download the following datasets that we utilized during the fine-tuning. The location of the datasets in the file directory is described in ***File Structure***.
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DiFF](https://github.com/xaCheng1996/DiFF)
### File Structure
The following is the **recommended file structure**. The files in each directory are described in the comments. You can customize the file structure by changing the corresponding directory information in `/datasets/finetune/preprocess_dlib/config/default.py`.
```bash
datasets/
├── DiFF/    # DiFF
│   ├── DiFF_real/ 				
│   ├── val/
│   │   ├── FS/
│   │   ├── FE/
│   │   ├── I2I/
│   │   └── T2I/ 				
│   ├── test/ 
│   │   ├── FS/
│   │   ├── FE/
│   │   ├── I2I/
│   │   └── T2I/ 			
│   ├── DiFF_val_faces/    # data after frame extraction (automatic creating)
│   └── DiFF_test_faces/    # data after frame extraction (automatic creating)
│
├── FaceForensics/    # FF++
│   ├── dataset/    # FF++ data partition
│   │   └── splits/ 
│   │       ├── train.json
│   │       ├── val.json
│   │       └── test.json
│   ├── original_sequences/    # FF++ real data
│   │  	└── youtube/
│   ├── manipulated_sequences/    # FF++ fake data
│   │   ├── DeepFakes/
│   │   ├── Face2Face/
│   │   ├── FaceSwap/
│   │   └── NeuralTextures/
│   └──face_dataset_split/    # data after frame extraction (automatic creating)
│
└── finetune/preprocess_dlib/
    ├── config/
    │   ├── __init__.py
    │   └── default.py    # define file structure
    ├── tools/
    │   └── util.py
    └── dataset_preprocess.py    # for frame extraction
```
### Frame Extraction
Run the program `/datasets/finetune/preprocess_dlib/dataset_preprocess.py` to extract video frames/people faces:
```python
python dataset_preprocess.py --dataset FF++
python dataset_preprocess.py --dataset DiFF
```
### Fine-tuning
Run the program `/FSFM_C3/main_finetune_DiFF.py` to fine-tune the model:
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=2 main_finetune_DiFF.py \
    --accum_iter 1 \
    --normalize_from_IMN \
    --apply_simple_augment \
    --batch_size 256 \
    --nb_classes 2 \
    --model vit_base_patch16 \
    --finetune [your pretained model ckpt] \
    --epochs 50 \
    --blr 5e-4 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval \
    --data_path /datasets/FaceForensics/face_dataset_split/dlib_no_align_no_resize/32_frames/DS_FF++_each_cls/c23/DeepFakes \
    --val_data_path /datasets/DiFF/DiFF_val_faces \
    --output_dir [path to save your model ckpt and logs]
```
### Evaluation
Run the program `/FSFM_C3/main_test_DiFF.py` to evaluate the model:
```python
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=2 main_test_DiFF.py \
    --normalize_from_IMN \
    --apply_simple_augment \
    --eval \
    --resume [your fine-tuned model ckpt] \
    --model vit_base_patch16 \
    --nb_classes 2 \
    --batch_size 256
```
## Task 3: FAS Detection
To evaluate the transferability of our method for FAS under significant domain shifts, we use four widely used benchmark datasets: **MSU-MFSD (M)**, **CASIA-FASD (C)**, **Idiap Replay-Attack (I)**, and **OULU-NPU (O)**. We treat each dataset as the target domain and apply the leave-one-out (LOO) cross-domain evaluation. 
### Dataset Preparation
Follow [this work](https://github.com/Kartik-3004/hyp-oc) to download the datasets that we utilized in this task. 
### File Structure
The following is the **recommended file structure**. The files in each directory are described in the comments.
```bash
FAS/
├── data/MCIO/
│   ├── frame/
│   │	├── casia/	# CASIA-FASD (C)
│   │	├── msu/	# MSU-MFSD (M)
│   │	├── oulu/	# OULU-NPU (O)
│   │	└── replay/	# Idiap Replay-Attack (I)
│   └── txt/			
│
├── config.py
├── fas.py
├── models_vit.py
├── train_vit.py
└── run_train_dlib_VGG_400_MCIO.sh
```
### Evaluation
Run the program `/FAS/run_train_dlib_VGG_400_MCIO.sh` to evaluate the model:

```bash
bash run_train_dlib_VGG_400_MCIO.sh
```
