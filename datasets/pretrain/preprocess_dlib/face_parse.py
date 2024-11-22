# -*- coding: utf-8 -*-

import sys
import torch
from PIL import Image
import numpy as np
from config import cfg
from tools.facer import facer
import os
import cv2
from PIL import Image
import random
from tqdm import tqdm
import shutil
import multiprocessing as mp
import torch.multiprocessing
import glob
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def vis_parsing_maps(im, parsing_anno, save_path=None):
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)
    # print(num_of_class) # 10

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.6, vis_parsing_anno_color, 0.6, 0)

    cv2.imwrite(save_path[:-4] + '.png', vis_parsing_anno_color)
    cv2.imwrite(save_path.replace('.png', '_with_pm.png'), vis_im)


def face_parsing(face_ds_path,
                 parsing_result_path,
                 face_img_dir=cfg.face_img_dir,
                 face_pm_dir=cfg.face_pm_dir,
                 save_vis_ps=False,
                 vis_pm_dir=cfg.vis_pm_dir):
    if device == 'cuda':
        gpu_id = torch.cuda.current_device()
        print(f"Process running on GPU {gpu_id}")

    face_detector = facer.face_detector('retinaface/mobilenet', device=device, threshold=0.3)  # 0.3 for FF++
    face_parser = facer.face_parser('farl/lapa/448', device=device)  # celebm parser

    face_img_path = os.path.join(parsing_result_path, face_img_dir)
    os.makedirs(face_img_path, exist_ok=True)
    face_pm_path = os.path.join(parsing_result_path, face_pm_dir)
    os.makedirs(face_pm_path, exist_ok=True)
    if save_vis_ps:
        vis_pm_path = os.path.join(parsing_result_path, vis_pm_dir)
        os.makedirs(vis_pm_path, exist_ok=True)

    with torch.inference_mode():
        no_parsing_num = 0
        for subdir, dirs, files in os.walk(face_ds_path):
            for file in tqdm(files):
                if file[-4:] == '.jpg' or '.png':
                    img = Image.open(os.path.join(subdir, file))
                    img = img.resize((cfg.face_size, cfg.face_size), Image.BICUBIC)
                    image = torch.from_numpy(np.array(img.convert('RGB')))
                    image = image.unsqueeze(0).permute(0, 3, 1, 2).to(device=device)
                    try:
                        faces = face_detector(image)
                        faces = face_parser(image, faces)

                        seg_logits = faces['seg']['logits']
                        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
                        seg_probs = seg_probs.data  # torch.Size([1, 11, 224, 224])
                        parsing = seg_probs.argmax(1)  # [1, 224, 224]

                        parsing_map = parsing.data.cpu().numpy()  # [1, 224, 224] int64
                        parsing_map = parsing_map.astype(np.int8)  # smaller space
                        # print(np.unique(parsing_map)) # [ 0  1  2  3  4  5  6  7  8  9 10]

                        file = file[:-4] + cfg.img_format
                        img.save(os.path.join(face_img_path, file))
                        save_path = os.path.join(face_pm_path, file).replace(cfg.img_format, cfg.pm_format)
                        np.save(save_path, parsing_map)
                        if save_vis_ps:
                            vis_parsing_maps(img, parsing_map.squeeze(0), save_path=os.path.join(vis_pm_path, file))

                    except KeyError:
                        no_parsing_num += 1
                        # parsing_map = np.zeros((1, 224, 224), dtype="int64")  # [1, 224, 224]

        print('no_parsing_face_num', no_parsing_num)


def start_process(rank, path, parsing_result_path):
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Process running on GPU {torch.cuda.current_device()}")
    face_parsing(face_ds_path=path, parsing_result_path=parsing_result_path)


def get_args_parser():
    parser = argparse.ArgumentParser('FSFM_C3 face parsing', add_help=False)
    parser.add_argument('--dataset', default='FF++',
                        help="choose from ['FF++', 'YTF', 'VF2']")

    return parser


if __name__ == '__main__':
    # extract face parsing map (.npy file) and vis imgs for pretraining:
    args = get_args_parser()
    args = args.parse_args()
    if args.dataset == 'FF++':
        # FF++_youtube:
        for i, path in enumerate(cfg.FF_real_face_paths_for_parsing):
            face_parsing(face_ds_path=path, parsing_result_path=cfg.FF_face_parse_ds_path, save_vis_ps=False)
    elif args.dataset == 'YTF':
        # YoutubeFace:
        face_parsing(face_ds_path=cfg.YTFace_path_for_parsing, parsing_result_path=cfg.YTFace_parse_ds_path, save_vis_ps=False)
    elif args.dataset == 'VF2':
        # VGGFace2:
        face_parsing(face_ds_path=cfg.VGGFace2_path_for_parsing, parsing_result_path=cfg.VGGFace2_parse_ds_path, save_vis_ps=False)
    else:
        print('Invalid Input')
    
    # torch.multiprocessing.set_start_method('spawn')
    # processes = []
    # for i, path in enumerate(cfg.FF_real_face_paths_for_parsing):
    #     p = mp.Process(target=start_process, args=(i, path))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()