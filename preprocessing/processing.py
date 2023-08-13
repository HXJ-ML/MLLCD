import os
import imageio
import math
import numpy as np
from PIL import Image
from tqdm import trange

OPENSLIDE_PATH = r'C:\ProgramData\Anaconda3\Library\openslide-win64-20220811\bin'

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


# Cut the pathological images into patches of the same size
def cut_patch(filename, patch_size):
    slide = openslide.open_slide(filename)
    w_count = int(slide.level_dimensions[0][0] // patch_size)
    h_count = int(slide.level_dimensions[0][1] // patch_size)

    # For saving useful patches
    # img_tensor = torch.tensor([], dtype=torch.float).to(DEVICE)
    img_array = np.array(slide.read_region((0,0), level=0, size=(patch_size, patch_size)), dtype=np.uint8)[:, :, 0:3]
    # Record the number of valid patches
    count = 0
    # delete useless patches
    for w in trange(w_count):
        for h in range(h_count):
            patch = np.array(
                slide.read_region((w * patch_size, h * patch_size), level=0, size=(patch_size, patch_size)),
                dtype=np.uint8)[:, :, 0:3]
            a = np.mean(patch)
            # print(a)
            # Set a threshold, if sum less than threshold, the patch is valuable.
            if a <= 220:
                img_array = np.concatenate([img_array, patch], axis=0)
                count = count + 1
    # print(count)
    img_array = img_array[patch_size:,:,:]
    # print(img_array.shape)
    return img_array


def bilinear(img, save_path, des_w=1, des_h=1):
    src_w, src_h,_ = img.shape
    des_img = np.zeros((des_w, des_h, 3))
    scale_w = des_w / src_w
    scale_h = des_h / src_h
    for i in trange(des_w):
        for j in range(des_h):
            src_x = (i + 0.5) / scale_w - 0.5
            src_y = (j + 0.5) / scale_h - 0.5
            x1 = math.floor(src_x)
            x2 = x1 + 1
            y1 = math.floor(src_y)
            y2 = y1 + 1
            if 0 <= x1 < src_w - 1 and 0 <= y1 < src_h - 1:
                des_img[i, j, 0] = (y2 - src_y) * (x2 - src_x) * img[x1, y1, 0] + (y2 - src_y) * (src_x - x1) * img[
                    x2, y1, 0] + (src_y - y1) * (x2 - src_x) * img[x1, y2, 0] + (src_y - y1) * (src_x - x1) * img[
                                       x2, y2, 0]
                des_img[i, j, 1] = (y2 - src_y) * (x2 - src_x) * img[x1, y1, 1] + (y2 - src_y) * (src_x - x1) * img[
                    x2, y1, 1] + (src_y - y1) * (x2 - src_x) * img[x1, y2, 1] + (src_y - y1) * (src_x - x1) * img[
                                       x2, y2, 1]
                des_img[i, j, 2] = (y2 - src_y) * (x2 - src_x) * img[x1, y1, 2] + (y2 - src_y) * (src_x - x1) * img[
                    x2, y1, 2] + (src_y - y1) * (x2 - src_x) * img[x1, y2, 2] + (src_y - y1) * (src_x - x1) * img[
                                       x2, y2, 2]

    img_new = Image.fromarray(np.uint8(des_img))
    imageio.imsave(save_path, img_new)


PATCH_SIZE = 1024
DES_PATCH_NUM = 60

# Data root path
root = '/data/hxjdata/svsdata'
# DataSet type
dataSet_list = ['CCRCC', 'CM', 'PDA', 'SAR', 'UCEC', 'LSCC', 'LUAD']
# Support set or query set
s_or_q_list = ['SupportSet_5', 'SupportSet_10', 'SupportSet_20', 'SupportSet_100', 'QuerySet']
# abnormal or normal
class_list = ['abnormal', 'normal']

for k in range(6,7):
    for i in range(0,1):
        print(dataSet_list[k], class_list[i] )
        dir = os.path.join(root, dataSet_list[k], 'wsi')
        path = os.path.join(dir,class_list[i])
        file_list = os.listdir(path)
        for m in range(0, 150):
            print(m, file_list[m])
            file = os.path.join(path, str(file_list[m]))
            save_filename = os.path.join(dir, class_list[i] + '_png', str(file_list[m][0:12]) + '.png')
            isExists = os.path.exists(save_filename)
            if not isExists:
                # print(save_filename)
                print('------------ cutting patches --------')
                image = cut_patch(filename=file, patch_size=PATCH_SIZE)
                # Save new pictures
                print('---------- saving ---------------')
                bilinear(img=image, save_path=save_filename, des_w=PATCH_SIZE * DES_PATCH_NUM, des_h=PATCH_SIZE)

