import numpy as np
import cv2
import glob
import os
import scipy.ndimage
import time


def main():
    img_dir = 'anime_face_data'
    fg_dir = 'mask_out_data'
    os.makedirs(fg_dir, exist_ok=True)
    mask_dir = 'masks_data'
    os.makedirs(mask_dir, exist_ok=True)

    img_lists = glob.glob(os.path.join(img_dir, '*.jpg'))

    t = time.time()

    for img_fn in img_lists:
        print("processing {} ....".format(img_fn))

        img = cv2.imread(img_fn)
        img = cv2.resize(img, (512, 512))

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask = np.zeros(img.shape[:2], np.uint8)
        mask_out = np.array(np.ones_like(img)*255).astype(np.uint8)

        rect = (1, 1, 510, 510)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        mask2 = scipy.ndimage.morphology.binary_fill_holes(mask2)

        img = img * mask2[:, :, np.newaxis]
        mask_out = mask_out * mask2[:, :, np.newaxis]

        mask_out_fn = img_fn.replace(img_dir, fg_dir)
        mask_fn = img_fn.replace(img_dir, mask_dir)

        cv2.imwrite(mask_out_fn, img)
        cv2.imwrite(mask_fn, mask_out)
        # break

    time_used = (time.time() - t) * 1.0 / len(img_lists)
    print('time used: ', time_used)  # time used:  2.1260950350761414


if __name__ == '__main__':
    main()
