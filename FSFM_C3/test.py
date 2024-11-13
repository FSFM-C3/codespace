# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# BYOL: https://github.com/lucidrains/byol-pytorch
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
from pathlib import Path
import submitit

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lr_sched import cosine_scheduler
from util.datasets import get_mean_std, FaceParsingDataset, collate_fn_crfrp
from util.pos_embed import interpolate_pos_embed, interpolate_pos_embed_ema

import models_fsfm
from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('FSFM_C3 pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='fsfm_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='patch size')

    parser.add_argument('--normalize_from_IMN', action='store_true',
                        help='Normalize(use mean/std) from ImageNet, else from pretrain datasets')
    parser.add_argument('--apply_simple_augment', action='store_true',
                        help='apply MAE simple data augment(ramdom-size crop and flip)')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--weight_sfr', default=0.007, type=float,
                        help='weight of additional loss(L_rec_fr) in specific facial region reconstruction.')
    parser.add_argument('--cl_loss', default='SimSiam', type=str,
                        help='contrastive loss function: SimSiam, BYOL, InfoNCE(or choose: MOCOv3、SimCLR）')
    parser.add_argument('--cl_sample', default='all', type=str,
                        help='apply for InfoNCE to make it negative-free '
                             'contrastive loss on all(original) or positive-only(no-negative) sample: all, positive')
    parser.add_argument('--weight_cl', default=0.1, type=float,
                        help='weight of contrastive learning loss (default: 0.1).')

    parser.add_argument('--t_momentum', type=float, default=0.996, metavar='M',
                        help='teacher momentum (default: 0.996) for EMA')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    # parser.add_argument('--data_path', default='./datasets/pretrain_datasets/VGGFace2/', type=str,
    #                     help='dataset path')
    parser.add_argument('--pretrain_data_path', dest='data_path', action='append', default=[],
                        help='support pretraining from multi datasets: use multi --pretrain_data_path could get list of dataset paths')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for default: ./checkpoint/{user}/')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log, empty for default: ./checkpoint/{user}/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--resume_target_encoder', default='',
                        help='resume target_network from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    cur_file_path = Path("checkpoint").absolute()
    cur_file_path.mkdir(parents=True, exist_ok=True)
    if Path("checkpoint/").is_dir():
        p = Path(f"{cur_file_path}/{user}/experiments_pretrain")
        p.mkdir(parents=True, exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def main(args):
    log_detail = 'log_detail' + '.txt'
    # sys.stdout = open(os.path.join(args.output_dir, log_detail), 'a+')

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.normalize_from_IMN:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        # # uncomment to use VGGFace2:
        # mean = [0.5482207536697388, 0.42340534925460815, 0.3654651641845703]
        # std = [0.2789176106452942, 0.2438540756702423, 0.23493893444538116]

        # to use FF++_youtube(faces from all  train and val frames):
        mean = [0.532625138759613, 0.4048449993133545, 0.3708747327327728]
        std = [0.25850796699523926, 0.21054500341415405, 0.20785294473171234]
        ds_stat = {'mean': mean, 'std': std}

        # #  from pretraining datasets:
        # mean, std = get_mean_std(args)
        # ds_stat = {'mean': mean.tolist(), 'std': std.tolist()}

        with open(os.path.join(args.output_dir, "pretrain_ds_mean_std.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(ds_stat) + "\n")

    if args.apply_simple_augment:
        # simple augmentation in ori MAE
        transform_train = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        # no data augmentation
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=mean, std=std)])

    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'pretrain'), transform=transform_train)
    if len(args.data_path) > 1:
        pretrain_datasets = [FaceParsingDataset(root=os.path.join(path), transform=transform_train) for path in
                             args.data_path]
        dataset_train = ConcatDataset(pretrain_datasets)
    else:
        pretrain_dataset = args.data_path[0]
        dataset_train = FaceParsingDataset(root=os.path.join(pretrain_dataset), transform=transform_train)

    collate_fn = collate_fn_crfrp(input_size=args.input_size,
                                  patch_size=args.patch_size,
                                  mask_ratio=args.mask_ratio)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        print(f'[INFO]log dir: %{args.log_dir}')
        # Writes entries directly to event files in the log_dir to be consumed by TensorBoard.
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    # import multiprocessing
    # args.num_workers = multiprocessing.cpu_count()
    # print(args.num_workers)

    for num_workers in range(0, 50, 10):
        print(num_workers)
        dl = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=collate_fn
        )

        start = time.time()
        for i, data in enumerate(dl, 0):
            pass
        end = time.time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

        del dl

    exit(0)


if __name__ == '__main__':
    # args = get_args_parser()
    # args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    #
    # main(args)

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir == '':
        print(os.getpgrp())
        # args.output_dir = get_shared_folder() / "%j"
        args.output_dir = get_shared_folder() / str(os.getpgrp())
    args.log_dir = args.output_dir

    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)
    executor.update_parameters(name="mae")
    job = executor.submit(main(args))
    print("Submitted job_id:", job.job_id)
