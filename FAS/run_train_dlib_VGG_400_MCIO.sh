#!/usr/bin/python

#--------FSFM-C3(VGGFace2-PT)---#
##############CIO2M==============
CUDA_VISIBLE_DEVICES=2 python train_vit.py \
    --pt_model /checkpoints/pt_models/VGGFace2/checkpoint-400.pth \     # path to your pretrained model ckpt
    --op_dir results/M/VGGFace2/400 \
    --report_logger_path results/M/VGGFace2/400/performance.csv \
    --config M \

##############MIO2C==============
CUDA_VISIBLE_DEVICES=2 python train_vit.py \
    --pt_model /checkpoints/pt_models/VGGFace2/checkpoint-400.pth \     # path to your pretrained model ckpt
    --op_dir results/C/VGGFace2/400 \
    --report_logger_path results/C/VGGFace2/400/performance.csv \
    --config C \

##############MCO2I==============
CUDA_VISIBLE_DEVICES=2 python train_vit.py \
    --pt_model /checkpoints/pt_models/VGGFace2/checkpoint-400.pth \     # path to your pretrained model ckpt
    --op_dir results/I/VGGFace2/400 \
    --report_logger_path results/I/VGGFace2/400/performance.csv \
    --config I \

##############MCI2O==============
CUDA_VISIBLE_DEVICES=2 python train_vit.py \
    --pt_model /checkpoints/pt_models/VGGFace2/checkpoint-400.pth \     # path to your pretrained model ckpt
    --op_dir results/O/VGGFace2/400 \
    --report_logger_path results/O/VGGFace2/400/performance.csv \
    --config O \