import numpy as np


def bpDice(pd, gt, label):
    bp_pd = pd == label
    bp_gt = gt == label
    overlap = np.sum(np.logical_and(bp_pd, bp_gt))
    union = np.sum(bp_pd) + np.sum(bp_gt)
    return 2*overlap/union


def mulDice(pd, gt):
    pd_onehot = (np.arange(gt.max()+1)==pd[..., None]).astype(int)
    # deleting background pixel
    pd_onehot[:, :, 0] = 0
    gt_onehot = (np.arange(gt.max()+1)==gt[..., None]).astype(int)
    # deleting background pixel
    gt_onehot[:, :, 0] = 0
    overlap = np.sum(pd_onehot*gt_onehot)
    union = np.sum(pd_onehot) + np.sum(gt_onehot)
    return 2*overlap/union


