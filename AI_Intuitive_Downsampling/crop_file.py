import glob
import os.path

import PIL.Image as Image
import tqdm

import multiprocessing
import time

## 768->672

def run_mp(img_path, dst_path):
    img = Image.open(img_path)
    img = img.crop((48, 48, 720, 720))
    img.save(dst_path)

def deamon_thread(q):
    print("I love work!")
    while not q.empty():
        img_path, dst_path = q.get()
        run_mp(img_path, dst_path)
        print(q.qsize())
    print("Bye~")


if __name__ == '__main__':
    dst_root = '/media/wzh/wxli/Camelyon16_672/train/tumor/'
    img_list = glob.glob('/media/wzh/wxli/Camelyon16_Trans/train/tumor/*png')
    # print(len(img_list))
    q = multiprocessing.Queue()

    for img_path in tqdm.tqdm(img_list):
        name = os.path.basename(img_path)
        q.put([img_path, os.path.join(dst_root, name)])

    for i in range(100):
        p = multiprocessing.Process(target=deamon_thread, args=(q,))
        p.start()
        time.sleep(1)