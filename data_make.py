import multiprocessing as mp
import sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
import numpy as np
import scipy
import os
import glob
import scipy.ndimage
import cv2.saliency
import time

img_dir = 'anime_face_traindata'
fg_dir = 'mask_out_traindata'
mask_dir = 'masks_traindata'

os.makedirs(img_dir, exist_ok=True)
os.makedirs(fg_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

saliency = cv2.saliency.StaticSaliencyFineGrained_create()


def get_circle(r=30):
    """
    simply generate a circle mask
    """
    image = np.zeros((2*r, 2*r, 3), np.uint8)
    cv2.circle(image,(r,r),r,(255,255,255),-1)
    white = np.array([255, 255, 255])
    mask = image[:, :, :] == white[np.newaxis, np.newaxis, :]
    mask = np.mean(mask, axis=2)
    return mask.astype(np.uint8)

circle_mask = get_circle()


def largest_cc(mask):
    """
    only keep the largest connected component for final mask
    """
    cc_out = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8, cv2.CV_32S)
    num_ccs = cc_out[0]
    labels = cc_out[1]
    stats = cc_out[2]

    max_area = 0
    max_label = 0
    for n in range(1, num_ccs):
        fill = scipy.ndimage.morphology.binary_fill_holes(labels == n).astype(np.float32)

        labels = labels + (fill - (labels == n)) * n
        stats[n, cv2.CC_STAT_AREA] = np.count_nonzero(labels == n)
        if stats[n, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[n, cv2.CC_STAT_AREA]
            max_label = n
    lcc = (labels == max_label)
    return lcc.astype(np.uint8)


def img_cut(img):
    """
    delete the black edge of an image
    """
    img2d = np.mean(img, axis=2)

    img2d_ud = np.mean(img2d, axis=1)
    img2d_lr = np.mean(img2d, axis=0)
    u, d = 0, img2d.shape[0]
    l, r = 0, img2d.shape[1]

    while u < img2d.shape[0] and img2d_ud[u] <= 10: u += 1
    while d > 0 and img2d_ud[d-1] <= 10: d -= 1
    while l < img2d.shape[1] and img2d_lr[l] <= 10: l += 1
    while r > 0 and img2d_lr[r-1] <= 10: r -= 1

    return img[u:d, l:r, :], [u, l, d, r]


def proc_fn(img_fn):
    """
    worker for get an avatar foreground
    """
    print("processing {} ....".format(img_fn))

    """ 1. resize and cut the black border """
    img = cv2.imread(img_fn)
    raw_img = cv2.resize(img.copy(), (512, 512))
    img = cv2.resize(img, (128, 128))
    img, [u,l,d,r] = img_cut(img)

    """ 2. graph cut: naive procedure """
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.zeros(img.shape[:2], np.uint8)

    rect = [1, 1, img.shape[1]-1, img.shape[0]-1]
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)

    """ 2. graph cut: add a head prior """
    c0, c1 = mask.shape[0] // 2, mask.shape[1] // 2
    radius = 30
    head_prior = mask[c0-radius:c0+radius, c1-radius:c1+radius]
    head_prior[circle_mask==1] = 1
    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask2 = largest_cc(mask2)

    """ 3. convert mask back to original image space"""
    mask2_ = np.zeros(shape=(128, 128))
    mask2_[u:d, l:r] = mask2
    mask2 = mask2_
    mask2 = cv2.resize(mask2, (raw_img.shape[1], raw_img.shape[0]))

    img = raw_img * mask2[:, :, np.newaxis]
    mask_out = np.array(np.ones_like(raw_img) * 255).astype(np.uint8)
    mask_out = mask_out * mask2[:, :, np.newaxis]

    mask_out_fn = img_fn.replace(img_dir, fg_dir)
    mask_fn = img_fn.replace(img_dir, mask_dir)

    cv2.imwrite(mask_out_fn, img)
    cv2.imwrite(mask_fn, mask_out)


def download(id):
    os.system('wget -P {} http://thiswaifudoesnotexist.net/example-{}.jpg'.format(img_dir, id))


if __name__ == '__main__':
    time_stat = True

    cpu_count = mp.cpu_count()

    if time_stat:
        cpu_count = 1

    pool = mp.Pool(cpu_count)
    # id_lists = list(range(41700, 41801))
    # pool.map(download, id_lists)
    #
    img_lists = glob.glob(os.path.join(img_dir, '*.jpg'))

    t = time.time()
    pool.map(proc_fn, img_lists)
    ave_t = (time.time() - t) / len(img_lists)
    print('time consumption of one image processing:', ave_t) # 0.132s

    # img_fn = './anime_face_traindata/example-41653.jpg'
    # proc_fn(img_fn)

