import random
import numpy as np
from skimage.filters import gaussian
import torch
from PIL import Image, ImageFilter
import base64
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class FreeScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(reversed(size))  # size: (h, w)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))


class RandomGaussianBlur(object):
    def __call__(self, img):
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))


def im2vl(img):
    img_tmp = np.zeros(img.shape[:2], dtype=np.uint8)
    head_msk = np.all(img == [255, 255, 255], axis=2)
    hand_msk = np.all(img == [255, 0, 0], axis=2)
    body_msk = np.all(img == [0, 255, 0], axis=2)
    foot_msk = np.all(img == [0, 0, 255], axis=2)
    img_tmp[head_msk] = 1
    img_tmp[hand_msk] = 2
    img_tmp[body_msk] = 3
    img_tmp[foot_msk] = 4
    return img_tmp


def vl2im(img):
    img_tmp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    head_msk = img == 1
    hand_msk = img == 2
    body_msk = img == 3
    foot_msk = img == 4
    img_tmp[head_msk] = [255, 255, 255]
    img_tmp[hand_msk] = [255, 0, 0]
    img_tmp[body_msk] = [0, 255, 0]
    img_tmp[foot_msk] = [0, 0, 255]
    return img_tmp


def vl2ch(img):
    h = img.shape[0]
    w = img.shape[1]
    img_tmp = np.zeros((h, w, 5), dtype=np.uint8)
    for c in range(5):
        img_tmp[:, :, c] = (img == c).astype(np.uint8)
    return img_tmp


def base64tonpy(file):
    base64_decoded = base64.b64decode(file)
    im_arr = np.frombuffer(base64_decoded, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img[:, :, ::-1]


def resize(im, desired_size, interpolation):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation=interpolation)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im

def CRF(img, msk):
    anno_rgb = msk.astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    colors, labels = np.unique(anno_lbl, return_inverse=True)

    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    n_labels = len(set(labels.flat))

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    # gt_prob: The certainty of the ground-truth (must be within (0,1)).
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=(10, 10), srgb=(5, 5, 5), rgbim=img, compat=5,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)
    MAP = np.argmax(Q, axis=0)
    MAP = colorize[MAP, :]
    return MAP.reshape(img.shape)
