import glob
import os
import tqdm
import PIL.ImageFile
import PIL.Image
import multiprocessing
import time

def deamon_thread(q):
    print("I love work!")
    while not q.empty():
        img_path = q.get()
        name = os.path.basename(img_path)
        if 'test' not in img_path:
            continue
        print(name)
        tag = name.split('.')[-2].split('_')[-1]
        # if tag not in ['01', '06', '11', '16', '21', '26']:
        #     continue
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        PIL.Image.MAX_IMAGE_PIXELS = None

        img = PIL.Image.open(img_path)
        size = img.size
        img = img.resize((int(size[0]*0.9), int(size[1]*0.9)))
        # img = img.resize((int(size[0]//2), int(size[1]//2)))
        if os.path.exists(os.path.join(visualization_root, name)):
            continue

        img.save(os.path.join(visualization_root, name))

    print("Bye~")

if __name__ == '__main__':
    q = multiprocessing.Queue()

    val_root = '/dssg/home/acct-eexdl/eexdl/liwenxi/slide_full_level_2/'
    condition_img_paths = glob.glob(os.path.join(val_root, '*.jpg'))
    condition_img_paths = condition_img_paths[::-1]
    print(os.path.join(val_root, '*.jpg'))
    visualization_root = './slide_full_level_2_0.9'
    if not os.path.exists(visualization_root):
        os.makedirs(visualization_root)

    count = 0
    for img_path in tqdm.tqdm(condition_img_paths):
        q.put(img_path)


    for i in range(5):
        p = multiprocessing.Process(target=deamon_thread, args=(q,))
        p.start()
        time.sleep(3)