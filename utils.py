import numpy as np
import math

# https://github.com/jianzhangcs/ISTA-Net-PyTorch
def my_zero_pad(img, block_size=32):
    old_h, old_w = img.shape
    delta_h = (block_size - np.mod(old_h, block_size)) % block_size
    delta_w = (block_size - np.mod(old_w, block_size)) % block_size
    img_pad = np.concatenate((img, np.zeros([old_h, delta_w])), axis=1)
    img_pad = np.concatenate((img_pad, np.zeros([delta_h, old_w + delta_w])), axis=0)
    new_h, new_w = img_pad.shape
    return img, old_h, old_w, img_pad, new_h, new_w

# https://github.com/jianzhangcs/ISTA-Net-PyTorch
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# https://github.com/cszn
def H(img, mode, inv=False):
    if inv:
        mode = [0, 1, 2, 5, 4, 3, 6, 7][mode]
    if mode == 0:
        return img
    elif mode == 1:
        return img.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(3, [2, 3])
    elif mode == 4:
        return img.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return img.rot90(1, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.rot90(3, [2, 3]).flip([2])