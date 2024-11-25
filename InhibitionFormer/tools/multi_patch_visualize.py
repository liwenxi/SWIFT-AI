from itertools import product
from math import ceil
from pathlib import Path

from mmdet.datasets import replace_ImageToTensor

import warnings
import glob
import os
import pickle
import tqdm

import mmcv
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.ops import nms

from argparse import ArgumentParser

import torch

import time

all_time = 0

def nmbs(bounding_boxes, scores, Nt):
    if len(bounding_boxes) == 0:
        return [], []
    bboxes = np.array(bounding_boxes)
    scores = np.array(scores)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    # order = np.argsort(scores)
    order = np.argsort(areas)
    picked_boxes = []
    while order.size > 0:
        index = order[-1]
        picked_boxes.append(bounding_boxes[index])
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h
        ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ious < Nt)
        order = order[left]
    return np.array(picked_boxes)

# def nms(bounding_boxes, Nt):
#     if len(bounding_boxes) == 0:
#         return [], []
#     bboxes = np.array(bounding_boxes)
#     x1 = bboxes[:, 0]
#     y1 = bboxes[:, 1]
#     x2 = bboxes[:, 2]
#     y2 = bboxes[:, 3]
#     scores = bboxes[:, 4]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = np.argsort(scores)
#     picked_boxes = []
#     while order.size > 0:
#         index = order[-1]
#         picked_boxes.append(bounding_boxes[index])
#         x11 = np.maximum(x1[index], x1[order[:-1]])
#         y11 = np.maximum(y1[index], y1[order[:-1]])
#         x22 = np.minimum(x2[index], x2[order[:-1]])
#         y22 = np.minimum(y2[index], y2[order[:-1]])
#         w = np.maximum(0.0, x22 - x11 + 1)
#         h = np.maximum(0.0, y22 - y11 + 1)
#         intersection = w * h
#         ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
#         left = np.where(ious < Nt)
#         order = order[left]
#     return picked_boxes

class PANDA(Dataset):
    def __init__(self, mode="train", **kwargs):
        self.root = "/home/wenxi/panda/images" if mode == "train" else \
                "/home/liwenxi/panda/images_test"
        temp = []
        self.paths = glob.glob(os.path.join(self.root, '*jpg'))
        self.paths.sort()
        self.gt_type = kwargs['gt_type']
        if mode == "train":
            for path in self.paths:
                name = os.path.basename(path)
                tag = name.split('.')[-2].split('_')[-1]
                if tag not in ['01', '06', '11', '16', '21', '26']:
                    temp.append(path)
        else:
            for path in self.paths:
                name = os.path.basename(path)
                tag = name.split('.')[-2].split('_')[-1]
                temp.append(path)
        self.paths = temp
        self.transform = kwargs['transform']
        self.length = len(self.paths)
        self.load_raw_img = kwargs['raw']
        # self.dataset = self.load_data()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.load_raw_img:
            img_path = self.paths[item]
            raw_path = img_path
            # raw_path = img_path
            raw = cv2.imread(raw_path)
            name = os.path.basename(img_path)
        # img, den = self.load_data(item)
        img, den = torch.rand(1), torch.rand(1)
        # if self.transform is not None:
        #     img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1).div(255)
        #     if len(den.shape) > 2:
        #         den = torch.tensor(den, dtype=torch.float).permute(2, 0, 1)
        #     else:
        #         den = torch.tensor(den, dtype=torch.float)
        #     img = self.transform(img)
        if self.load_raw_img:
            return img, den, raw, name
        return img, den

    def load_data(self, item):
        img_path = self.paths[item]
        if self.gt_type == 'adaptive_16':
            gt_path = img_path.replace('.jpg', '.h5').replace('images_1024', 'density_map_adaptive_16')
        elif self.gt_type == 'fixed_16':
            gt_path = img_path.replace('.jpg', '.h5').replace('images_1024', 'density_map_16')
        elif self.gt_type == 'adaptive_8':
            gt_path = img_path.replace('.jpg', '.h5').replace('images_1024', 'density_map_adaptive_8')
        elif self.gt_type == 'adaptive_4scale_16':
            gt_path = img_path.replace('.jpg', '.h5').replace('images_1024', 'density_map_adaptive_4scale_16')

        # img = skimage.io.imread(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(1024, 1024))

        gt_file = h5py.File(gt_path)
        den = np.asarray(gt_file['density'])
        # den = den[:, :, np.newaxis]

        den = cv2.resize(den, dsize=(1024, 1024))*(den.shape[0]*den.shape[1]/1024**2)
        # # Sum Density
        # for i in range(3, 0, -1):
        #     den[:, :, i] += den[:, :, i-1]
        # for i in range(1, 4):
        #     den[:, :, i] += den[:, :, i-1]

        #
        # h = den.shape[0]
        # w = den.shape[1]
        # h_trans = h // 8
        # w_trans = w // 8
        #
        # den = cv2.resize(den, (w_trans, h_trans),
        #                  interpolation=cv2.INTER_CUBIC) * (h * w) / (h_trans * w_trans)

        # print(img.shape, den.shape)
        return img, den

def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def get_multiscale_patch(sizes, steps, ratios):
    """Get multiscale patch sizes and steps.

    Args:
        sizes (list): A list of patch sizes.
        steps (list): A list of steps to slide patches.
        ratios (list): Multiscale ratios. devidie to each size and step and
            generate patches in new scales.

    Returns:
        new_sizes (list): A list of multiscale patch sizes.
        new_steps (list): A list of steps corresponding to new_sizes.
    """
    assert len(sizes) == len(steps), 'The length of `sizes` and `steps`' \
                                     'should be the same.'
    new_sizes, new_steps = [], []
    size_steps = list(zip(sizes, steps))
    for (size, step), ratio in product(size_steps, ratios):
        new_sizes.append(int(size / ratio))
        new_steps.append(int(step / ratio))
    return new_sizes, new_steps

def slide_window(width, height, sizes, steps, img_rate_thr=0.6):
    """Slide windows in images and get window position.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        sizes (list): List of window's sizes.
        steps (list): List of window's steps.
        img_rate_thr (float): Threshold of window area divided by image area.

    Returns:
        np.ndarray: Information of valid windows.
    """
    assert 1 >= img_rate_thr >= 0, 'The `in_rate_thr` should lie in 0~1'
    windows = []
    # Sliding windows.
    for size, step in zip(sizes, steps):
        size_w, size_h = size
        step_w, step_h = step

        x_num = 1 if width <= size_w else ceil((width - size_w) / step_w + 1)
        x_start = [step_w * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size_w > width:
            x_start[-1] = width - size_w

        y_num = 1 if height <= size_h else ceil((height - size_h) / step_h + 1)
        y_start = [step_h * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size_h > height:
            y_start[-1] = height - size_h

        start = np.array(list(product(x_start, y_start)), dtype=np.int64)
        windows.append(np.concatenate([start, start + size], axis=1))
    windows = np.concatenate(windows, axis=0)

    # Calculate the rate of image part in each window.
    img_in_wins = windows.copy()
    img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
    img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
    img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                (img_in_wins[:, 3] - img_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * \
                (windows[:, 3] - windows[:, 1])
    img_rates = img_areas / win_areas
    if not (img_rates >= img_rate_thr).any():
        img_rates[img_rates == img_rates.max()] = 1
    return windows[img_rates >= img_rate_thr]

def merge_results(results, offsets, iou_thr=0.6, device='cpu'):
    """Merge patch results via nms.

    Args:
        results (list[np.ndarray]): A list of patches results.
        offsets (np.ndarray): Positions of the left top points of patches.
        iou_thr (float): The IoU threshold of NMS.
        device (str): The device to call nms.

    Retunrns:
        list[np.ndarray]: Detection results after merging.
    """
    assert len(results) == offsets.shape[0], 'The `results` should has the ' \
                                             'same length with `offsets`.'
    print("HHafoufheu : ", iou_thr)
    merged_results = []
    for results_pre_cls in zip(*results):
        tran_dets = []
        for dets, offset in zip(results_pre_cls, offsets):
            dets[:, :2] += offset
            dets[:, 2:4] += offset
            tran_dets.append(dets)
        tran_dets = np.concatenate(tran_dets, axis=0)

        # #************
        # merged_results.append(tran_dets)
        # #************

        if tran_dets.size == 0:
            merged_results.append(tran_dets)
        else:
            tran_dets = torch.from_numpy(tran_dets)
            tran_dets = tran_dets.to(device)
            nms_dets, _ = nms(tran_dets[:, :4].contiguous(), tran_dets[:, -1].contiguous(),
                                      0.4)
            merged_results.append(nms_dets.cpu().numpy())
    return merged_results


def inference_detector_by_patches(model,
                                  img,
                                  sizes,
                                  steps,
                                  ratios,
                                  merge_iou_thr,
                                  bs=10):
    """inference patches with the detector.
    Split huge image(s) into patches and inference them with the detector.
    Finally, merge patch results on one huge image by nms.
    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray or): Either an image file or loaded image.
        sizes (list): The sizes of patches.
        steps (list): The steps between two patches.
        ratios (list): Image resizing ratios for multi-scale detecting.
        merge_iou_thr (float): IoU threshold for merging results.
        bs (int): Batch size, must greater than or equal to 1.
    Returns:
        list[np.ndarray]: Detection results.
    """

    # if isinstance(img, (list, tuple)):
    #     is_batch = True
    # else:
    #     img = [img]
    #     is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    cfg = cfg.copy()
    # set loading pipeline type
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    if not isinstance(img, np.ndarray):
        img = mmcv.imread(img)

    height, width = img.shape[:2]
    # sizes, steps = get_multiscale_patch(sizes, steps, ratios)
    # windows = slide_window(width, height, sizes, steps)
    # windows = slide_window(width, height, [(4096, 2048)], [(4096-500, 2048-500)])
    # windows = slide_window(width, height, [(2560*3, 1280*3)], [(2560*3-500, 1280*3-500)])
    # windows = slide_window(width, height, [(1024*6, 512*6)], [(1024*6-500, 512*6-500)])

    # windows = slide_window(width, height, [(6144, 3072)], [(6144-1000, 3072-1000)])
    windows = slide_window(width, height, [(2000*3, 1200*3)], [(2000*3-600, 1200*3-600)])

    # windows = slidmue_window(width, height, [(3333, 2000)], [(3333, 2000)])
    # windows = slide_window(width, height, [(2048, 1024)], [(200, 200)])

    results = []
    start = 0

    time_start = time.time()
    while True:
        # prepare patch data
        patch_datas = []
        if (start + bs) > len(windows):
            end = len(windows)
        else:
            end = start + bs
        for window in windows[start:end]:
            x_start, y_start, x_stop, y_stop = window
            # patch_width = x_stop - x_start
            # patch_height = y_stop - y_start
            patch = img[y_start:y_stop, x_start:x_stop]
            # prepare data

            data = dict(img=patch)

            data = test_pipeline(data)
            patch_datas.append(data)

        data = collate(patch_datas, samples_per_gpu=len(patch_datas))
        # just get the actual data from DataContainer
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            for m in model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        # forward the model
        with torch.no_grad():
            results.extend(model(return_loss=False, rescale=True, **data))

        if end >= len(windows):
            break
        start += bs
    global all_time
    all_time += (time.time()-time_start)
    # print(time.time()-time_start)
    results = merge_results(
        results,
        windows[:, :2],
        iou_thr=merge_iou_thr,
        device=device)
    return results

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('img_path', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[1024],
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[824],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0],
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.6,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

def main(args):
    all_result = []
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a huge image by patches

    # root = "/media/wzh/wxli/PANDA/images_test"
    root = "/home/liwenxi/panda/images_test"

    paths = glob.glob(os.path.join(root, '*jpg'))
    paths.sort()
    paths = ['/home/liwenxi/panda/images_test/IMG_16_17.jpg']

    from torchvision import transforms
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_test = PANDA(mode="test", transform=transform, raw=True, gt_type='adaptive_4scale_16')
    # print(dataset_test.__len__())
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

    for img in tqdm.tqdm(paths):
    # for img, density, raw, name in tqdm.tqdm(dataloader_test):
    #     img = raw.squeeze().numpy()
        result = inference_detector_by_patches(model, img, args.patch_sizes,
                                               args.patch_steps, args.img_ratios,
                                               args.merge_iou_thr)
    # for img in tqdm.tqdm(paths):
    #     result = inference_detector_by_patches(model, img, args.patch_sizes,
    #                                            args.patch_steps, args.img_ratios,
    #                                            args.merge_iou_thr)

        model.show_result(
            img,
            result,
            score_thr=0.3,
            show=True,
            wait_time=0,
            win_name='result',
            bbox_color='blue',
            text_color=(200, 200, 200),
            thickness=15,
            mask_color=None,
            out_file='lin.jpg')
        print(result)
        break

        all_result.append(result)
    print(all_time/len(all_result))
    with open('sparse0.1.pkl', 'wb') as f:
        pickle.dump(all_result, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)

    device = 'cuda:0'
    device = 'cpu'
    all_result = []
    outputs = mmcv.load('sparse_0.5.pkl')

    # print(outputs)
    for results in outputs:
        merged_results = []
        for tran_dets in results:
            tran_dets = torch.from_numpy(tran_dets)
            tran_dets = tran_dets.to(device)
            # nms_dets, _ = nms(tran_dets[:, :4].contiguous(), tran_dets[:, -1].contiguous(),
            #                           iou_thr=0.6)
            # merged_results.append(nms_dets.cpu().numpy())
            nms_dets = nmbs(tran_dets[:, :5].contiguous(), tran_dets[:, -1].contiguous(),
                                      0.7)
            # nms_dets = tran_dets

            merged_results.append(nms_dets)
            # break
        # print(nms_dets.shape)
        # all_result.append([nms_dets.cpu().numpy()])
        all_result.append(merged_results)
    # print(all_result)
    with open('sparse0.1nms.pkl', 'wb') as f:
        pickle.dump(all_result, f)