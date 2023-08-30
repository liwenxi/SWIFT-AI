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


try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

@torch.no_grad()
def main(config):
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    model.eval()
    logger.info(str(model))

    # if config.AMP_OPT_LEVEL != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    transform = build_transform(is_train=False, config=config)

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    model_without_ddp = model.module
    # checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    # checkpoint = torch.load('output/giga_patch16_stride4_window7_448/default/ckpt_epoch_100.pth', map_location='cpu')
    # print('output/giga_patch16_stride4_window7_448/default/ckpt_epoch_160.pth')
    # checkpoint = torch.load('output/giga2_patch16_stride4_window7_448/default/ckpt_epoch_104.pth', map_location='cpu')

    # checkpoint = torch.load('output/giga2_patch16_stride4_window7_448_panda/default/ckpt_epoch_222.pth', map_location='cpu')
    # checkpoint = torch.load('output/giga2_patch16_stride4_window7_448_dota/default/ckpt_epoch_208.pth', map_location='cpu')
    checkpoint = torch.load('output/giga2_resnet_stride2_window7_448_dota/default/ckpt_epoch_244.pth',
                            map_location='cpu')

    # checkpoint = torch.load('output/giga2_base_patch16_stride4_window7_448_panda_x0.5/default/ckpt_epoch_296.pth', map_location='cpu')

    # print('output/giga_patch16_stride6_window7_672/default/ckpt_epoch_98.pth')
    msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    # val_root = '/dssg/home/acct-eexdl/eexdl/liwenxi/slide_full_level_2/'
    # condition_img_paths = glob.glob(os.path.join(val_root, 'test*.jpg'))

    # val_root = '/home/wenxi/panda/images/'
    val_root = '/media/wzh/wxli/split_ms_dota/test/images/'
    # val_root = '/dssg/home/acct-eexdl/eexdl/liwenxi/panda/images'
    condition_img_paths = glob.glob(os.path.join(val_root, '*.png'))
    print(os.path.join(val_root, '*.jpg'))
    visualization_root = './probs_map_dota_temp_res'
    if not os.path.exists(visualization_root):
        os.makedirs(visualization_root)

    count = 0
    for img_path in tqdm.tqdm(condition_img_paths):
        name = os.path.basename(img_path)
        tag = name.split('.')[-2].split('_')[-1]
        # if tag not in ['01', '06', '11', '16', '21', '26']:
        #     continue
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        PIL.Image.MAX_IMAGE_PIXELS = None

        img = PIL.Image.open(img_path).convert('RGB')
        size = img.size
        # img = img.resize((size[0]//2, size[1]//2))
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # import matplotlib.image as mpimg
        # img = mpimg.imread(img_path)
        name = os.path.basename(img_path)
        img_tensor = transform(img)
        img_tensor = img_tensor.cuda().unsqueeze(0)

        # compute output
        time_t = time.time()
        output, attn = model(img_tensor)
        print(output.shape)
        print(time.time()-time_t)
        output = output.squeeze(0)
        # print(attn.shape)
        softmax = torch.nn.Softmax(dim=0)
        print(output.shape)
        output = softmax(output)[1, :, :]
        print(output.shape)
        output[output<0.5] = 0
        output[output>=0.5] = 1
        plt.imsave(os.path.join(visualization_root, name), output.cpu().numpy(), cmap=plt.cm.jet)
        np.save(os.path.join(visualization_root, name.replace('.png', '.npy')), output.cpu().numpy())
        # _, pred = output.topk(1, 1, True, True)
        # print(pred.shape)
        # if pred.item() == 1:
        #     shutil.copyfile(img_path, os.path.join(visualization_root, name))
        #     GRID_SIZE = 16
        #     img = np.array(img)
        #     height, width, channels = img.shape
        #     for x in range(0, width - 1, GRID_SIZE):
        #         if x//GRID_SIZE % 6 == 0 or x//GRID_SIZE % 6 == 1:
        #             cv2.line(img, (x, 0), (x, height), (255, 255, 255), 1, 1)
        #         else:
        #             pass
        #             # cv2.line(img, (x, 0), (x, height), (0, 0, 0, ), 1, 1)
        #     for y in range(0, height - 1, GRID_SIZE):
        #         if y//GRID_SIZE % 6 == 0 or y//GRID_SIZE % 6 == 1:
        #             cv2.line(img, (0, y), (width, y), (255, 255, 255), 1, 1)
        #         else:
        #             pass
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     cv2.imwrite(os.path.join(visualization_root, name), img)






if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
