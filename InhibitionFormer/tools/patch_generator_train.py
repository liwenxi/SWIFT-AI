import json
import multiprocessing
import os
import sys
from itertools import product
from math import ceil

import cv2
import numpy as np


class PatchGenerator(object):
    def __init__(self, info, type='normal', data_dir='/home/liwenxi/panda/raw/PANDA/image_train',
                 save_img_path='/home/liwenxi/panda/raw/PANDA/patches/s_6000x6000', save_json_path='./new.json'):
        self.data_dir = data_dir
        self.type = type
        self.save_img_path = save_img_path
        self.info = info
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.num = 0
        self.ob = []
        if type == 'center':
            self.patch_size = [16]
        elif type == 'normal':
            self.patch_size = [4, 8, 16]
        if not os.path.exists(save_img_path):
            os.mkdir(save_img_path)

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.info):
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(self.info)))
            sys.stdout.flush()
            width = self.info[json_file]['image size']['width']
            height = self.info[json_file]['image size']['height']

            img = cv2.imread(os.path.join(self.data_dir, json_file))

            if self.type == 'center':
                center_list = self.get_center(json_file, width, height, self.patch_size)
            elif self.type == 'normal':
                patch_num = self.patch_size[0]
                self.patch_w = width // patch_num
                self.patch_h = height // patch_num
                center_list = self.normal_center(self.patch_w, self.patch_h, patch_num)
            elif self.type == 'sw':
                center_list = self.slide_window(width, height, [(6000, 6000)], [(5000, 5000)])

            for patch_id, center_lf_point in enumerate(center_list):
                x, y, patch_w, patch_h = center_lf_point
                self.patch_w = patch_w
                self.patch_h = patch_h
                patch = self.crop(img, y, x, patch_h, patch_w)
                self.file_name = os.path.basename(json_file).replace('.jpg', '_' + str(patch_id + 1).zfill(4) + '.jpg')
                patch_person_count = 0
                for obj in self.info[json_file]['objects list']:
                    if obj['category'] not in ['person']:
                        continue
                    self.supercategory = obj['category']
                    if self.supercategory not in self.label:
                        self.categories.append(self.categorie())
                        self.label.append(self.supercategory)
                    obj = obj['rects']['full body']
                    x1 = float(obj['tl']['x'] * width)
                    y1 = float(obj['tl']['y'] * height)
                    w = float((obj['br']['x'] - obj['tl']['x']) * width)
                    h = float((obj['br']['y'] - obj['tl']['y']) * height)

                    box_x1 = (x1 - x)
                    box_x2 = box_x1 + w
                    box_y1 = (y1 - y)
                    box_y2 = box_y1 + h

                    box_x1, box_x2 = np.clip((box_x1, box_x2), 0, patch_w)
                    box_y1, box_y2 = np.clip((box_y1, box_y2), 0, patch_h)

                    if (box_y2 - box_y1) * (box_x2 - box_x1) < w * h / 2:
                        continue

                    self.bbox = [box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1]  # COCO 对应格式[x,y,w,h]
                    self.area = (box_x2 - box_x1) * (box_y2 - box_y1)
                    self.annotations.append(self.annotation())
                    self.annID += 1
                    patch_person_count += 1
                if patch_person_count > 0:
                    self.images.append(self.image())
                    self.num += 1
                    patch = cv2.resize(patch, dsize=(self.patch_w, self.patch_h))
                    cv2.imwrite(os.path.join(self.save_img_path, self.file_name), patch)

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self):
        image = {}
        image['height'] = self.patch_h
        image['width'] = self.patch_w
        image['id'] = self.num + 1
        image['file_name'] = self.file_name
        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        categorie['id'] = len(self.label) + 1  # 0 is background
        categorie['name'] = self.supercategory
        return categorie

    def annotation(self):
        annotation = {}

        annotation['segmentation'] = [list(map(float, self.getsegmentation()))]
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1

        annotation['bbox'] = self.bbox
        annotation['area'] = self.area
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getsegmentation(self):
        return [0]

    def mask2polygons(self):
        contours = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bbox = []
        for cont in contours[1]:
            [bbox.append(i) for i in list(cont.flatten())]
        return bbox

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)

    def get_params(self, h, w, th, tw):
        if w == tw and h == th:
            return 0, 0, h, w

        i = np.random.randint(0, h - th + 1, size=(1,)).item()
        j = np.random.randint(0, w - tw + 1, size=(1,)).item()

        return i, j

    def random_crop(self, img, output_size):
        h, w, _ = img.shape
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )
        i, j = self.get_params(h, w, th, tw)
        target = self.crop(img, i, j, th, tw)
        return target, j, i

    def crop(self, img, i, j, th, tw):
        target = img[i:i + th, j:j + tw, :]
        return target

    def get_center(self, json_file, width, height, patch_size):
        center_list = []
        for patch_num in patch_size:
            patch_w = width // patch_num
            patch_h = height // patch_num
            for obj in self.info[json_file]['objects list']:
                if obj['category'] != 'person':
                    continue
                self.supercategory = obj['category']
                if self.supercategory not in self.label:
                    self.categories.append(self.categorie())
                    self.label.append(self.supercategory)
                x1 = float(obj['rects']['full body']['tl']['x'] * width)
                y1 = float(obj['rects']['full body']['tl']['y'] * height)
                w = float((obj['rects']['full body']['br']['x'] - obj['rects']['full body']['tl']['x']) * width)
                h = float((obj['rects']['full body']['br']['y'] - obj['rects']['full body']['tl']['y']) * height)

                center_x = x1 + w // 2
                center_y = y1 + h // 2

                lt_x = int(center_x - patch_w // 2)
                lt_y = int(center_y - patch_h // 2)
                if 0 < lt_x < width - patch_w + 1 and 0 < lt_y < height - patch_h + 1:
                    center_list.append((lt_x, lt_y, patch_w, patch_h))
        return center_list

    def normal_center(self, patch_w, patch_h, num_patch=16):
        center_list = []

        for i in range(num_patch):
            for j in range(num_patch):
                lt_x = i * patch_w
                lt_y = j * patch_h
                center_list.append((lt_x, lt_y, patch_w, patch_h))

        return center_list

    def slide_window(self, width, height, sizes, steps, img_rate_thr=0.6):
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

            start = np.array(list(product(x_start, y_start)), dtype=int)
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
        windows = windows[img_rates >= img_rate_thr]
        return [(int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])) for box in windows]


def worker1(train_info):
    PatchGenerator(train_info, type='sw',
                   save_json_path='/home/liwenxi/panda/raw/PANDA/coco_json/train_6000x6000.json')


def worker2(val_info):
    PatchGenerator(val_info, data_dir='/home/liwenxi/panda/raw/PANDA/image_test', type='sw', save_json_path='/home/liwenxi/panda/raw/PANDA/coco_json/test_s4.json')


if __name__ == '__main__':
    file = open('/home/liwenxi/panda/raw/PANDA/image_annos/person_bbox_train.json', 'r', encoding='utf-8')
    info = json.load(file)

    train_info = []

    for num, item in enumerate(info):
        train_info.append(item)

    file = open('./person_bbox_test_real.json', 'r', encoding='utf-8')
    info = json.load(file)

    test_info = []
    for num, item in enumerate(info):
        test_info.append(item)

    # print(train_info)
    # print(val_info)
    train_info = {item: info[item] for item in train_info}
    # val_info = {item: info[item] for item in val_info}
    # test_info = {item: info[item] for item in test_info}

    p1 = multiprocessing.Process(target=worker1, args=(train_info,))
    # p2 = multiprocessing.Process(target=worker2, args=(test_info,))

    p1.start()
    # p2.start()

    p1.join()
    # p2.join()