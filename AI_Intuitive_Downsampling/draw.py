# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import glob
import os
import shutil
import time
import random
import argparse
import datetime

import PIL.Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import tqdm

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader, build_transform
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
import torch.nn.functional as F

@torch.no_grad()
def main():
    visualization_root = './draw'
    if not os.path.exists(visualization_root):
        os.makedirs(visualization_root)

    count = 0

    val_root = os.path.join('/ssd1_2T/wxli/Camelyon16_Trans/', 'val')
    condition_img_paths = glob.glob(os.path.join(val_root, 'tumor', '*.png'))
    print(condition_img_paths)

    for img_path in tqdm.tqdm(condition_img_paths):
        img = PIL.Image.open(img_path)
        name = os.path.basename(img_path)

        GRID_SIZE = 16
        img = np.array(img)
        img = img[160:-160 , 160:-160, :]
        # img = cv2.resize(img, (448, 448))
        height, width, channels = img.shape

        cv2.imwrite(os.path.join(visualization_root, name), img)


        #####
        x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        B, C, H, W = x.shape
        # print(x.shape)
        window_size = 16
        pad_l = pad_t = 0
        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, Hp, Wp = x.shape

        x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
        x = x[:, :, ::4, :, ::4, ]

        x = x.flatten(-2, -1).flatten(-3, -2).contiguous()
        # print(x.shape)

        B, C, H, W = x.shape
        window_size = 16*7
        pad_l = pad_t = 0
        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, Hp, Wp = x.shape

        x = x.squeeze(0).permute(1, 2, 0).numpy()
        # print(x.shape)
        # print(type(x))
        cv2.imwrite(os.path.join(visualization_root, name.replace('.png', '_mini.png')), x)
        # plt.imsave(os.path.join(visualization_root, name), x)
        #####
        # for x in range(0, width - 1, GRID_SIZE):
        #     if x//GRID_SIZE % 4 == 0 or x//GRID_SIZE % 4 == 1:
        #         cv2.line(img, (x, 0), (x, height), (255, 255, 255), 1, 1)
        #     else:
        #         pass
        #         # cv2.line(img, (x, 0), (x, height), (0, 0, 0, ), 1, 1)
        # for y in range(0, height - 1, GRID_SIZE):
        #     if y//GRID_SIZE % 4 == 0 or y//GRID_SIZE % 4 == 1:
        #         cv2.line(img, (0, y), (width, y), (255, 255, 255), 1, 1)
        #     else:
        #         pass
        #
        # for x in range(0, width - 1, GRID_SIZE):
        #     for y in range(0, height - 1, GRID_SIZE):
        #         if (x // GRID_SIZE % 4 == 0) and (y//GRID_SIZE % 4 == 0):
        #             cv2.line(img, (x, y), (x+GRID_SIZE, y), (0, 255, 0), 1, 1)
        #             cv2.line(img, (x, y+GRID_SIZE), (x + GRID_SIZE, y+GRID_SIZE), (0, 255, 0), 1, 1)
        #             cv2.line(img, (x, y), (x, y+GRID_SIZE), (0, 255, 0), 1, 1)
        #             cv2.line(img, (x+GRID_SIZE, y), (x + GRID_SIZE, y+GRID_SIZE), (0, 255, 0), 1, 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(visualization_root, name), img)


if __name__ == '__main__':
    main()
