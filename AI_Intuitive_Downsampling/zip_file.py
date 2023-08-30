import glob
import os
import zipfile

def old_version():
    normal_list = glob.glob('/media/wzh/wxli/Camelyon16_Trans/train/normal/*png')
    tumor_list = glob.glob('/media/wzh/wxli/Camelyon16_Trans/train/tumor/*png')

    normal_list = ['train/normal/'+os.path.basename(line)+'\t0\n' for line in normal_list]
    tumor_list = ['train/tumor/' + os.path.basename(line)+'\t1\n' for line in tumor_list]
    train_list = normal_list+tumor_list
    with open('/media/wzh/wxli/Camelyon16-Zip/train_map.txt', 'w') as f:
        f.writelines(train_list)


    normal_list = glob.glob('/media/wzh/wxli/Camelyon16_Trans/val/normal/*png')
    tumor_list = glob.glob('/media/wzh/wxli/Camelyon16_Trans/val/tumor/*png')

    normal_list = ['val/normal/'+os.path.basename(line)+'\t0\n' for line in normal_list]
    tumor_list = ['val/tumor/' + os.path.basename(line)+'\t1\n' for line in tumor_list]
    val_list = normal_list+tumor_list
    with open('/media/wzh/wxli/Camelyon16-Zip/val_map.txt', 'w') as f:
        f.writelines(val_list)

if __name__ == '__main__':
    train_list = []
    with zipfile.ZipFile('/dssg/home/acct-eexdl/eexdl/liwenxi/Camelyon16/val.zip') as f:
        files = f.namelist()  # namelist() 返回zip压缩包中的所有文件
        for item in files:
            if '.txt' in item:
                continue
            if 'tumor' in item:
                train_list.append(item+'\t1\n')
            elif 'normal' in item:
                train_list.append(item+'\t0\n')
            else:
                print("wrong")
                break
    print(len(train_list))
    with open('/dssg/home/acct-eexdl/eexdl/liwenxi/Camelyon16/val_map.txt', 'w') as f:
        f.writelines(train_list)

