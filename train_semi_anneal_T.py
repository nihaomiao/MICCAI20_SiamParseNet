# using semi-supervised learning with anneal temperature
# Frozen BN when set is_training as false
import sys
sys.path.append("/workspace/code/infant-project/SPN")  # todo set path to your code directory
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
from torch.utils import data
import numpy as np
import cv2
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import timeit
import math
from PIL import Image
from SegUtils.transforms import vl2im
from ProUtils.misc import Logger
from SPN.SPNet import SemiSPNet, resnet_feature_layers
from SPNUtils.dataset import SegTrainSemi
from ProUtils.misc import vl2ch
import random
import numpy.random
import sys

start = timeit.default_timer()
BATCH_SIZE = 20
TOPK = 20
MAX_EPOCH = 40
root_dir = '/data/GMS-data/hfn5052/spn-youtube-infant'  # todo the path to save training results
data_dir = "/data/youtube-infant-body-parsing"  # todo the path to training data
GPU = "7"
T = 0.4
base_dist = 2/3  # final p_2 = 1-base_dist
postfix = "-spn-semi-anneal-T%.2f-B%.2f" % (T, base_dist)
# todo the image mean of dataset
IMG_MEAN = np.array((101.84807705937696, 112.10832843463207, 111.65973036298041), dtype=np.float32)
INPUT_SIZE = '256, 256'
# todo the class distribution
CLASS_DISTRI = [3.28588037e+08, 2.47989400e+07, 2.46785900e+07, 5.26057730e+07, 4.05293800e+07]
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 5
POWER = 0.9
RANDOM_SEED = 1234
# todo the path to pretrained COCO dataset, which can be downloaded from
# https://drive.google.com/file/d/0BxhUwxvLPO7TVFJQU1dwbXhHdEk/view?resourcekey=0-7UxnHrm5eDCyvz2G35aKgA
RESTORE_MATNET_FROM = "/data/GMS-data/hfn5052/ExtractNet/MS_DeepLab_resnet_pretrained_COCO_init.pth"
RESTORE_SEGNET_FROM = "/data/GMS-data/hfn5052/ExtractNet/MS_DeepLab_resnet_pretrained_COCO_init.pth"
RESTORE_SPNET_FROM = ""
NUM_EXAMPLES_PER_EPOCH = int(3e4)
SAVE_PRED_EVERY = (MAX_EPOCH*NUM_EXAMPLES_PER_EPOCH//BATCH_SIZE)//5
SNAPSHOT_DIR = os.path.join(root_dir, 'snapshots' + postfix)
IMGSHOT_DIR = os.path.join(root_dir, 'imgshots' + postfix)
WEIGHT_DECAY = 0.0005
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(IMGSHOT_DIR, exist_ok=True)

LOG_PATH = SNAPSHOT_DIR + "/B"+format(BATCH_SIZE, "04d")+"E"+format(MAX_EPOCH, "04d")+".log"
sys.stdout = Logger(LOG_PATH, sys.stdout)
print(postfix)
print("RESTORE_PROPNET_FROM", RESTORE_MATNET_FROM)
print("RESTORE_SEGNET_FROM", RESTORE_SEGNET_FROM)
print("RESTORE_SPNET_FROM", RESTORE_SPNET_FROM)
print("num of epoch:", MAX_EPOCH)
print(NUM_EXAMPLES_PER_EPOCH)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SPN")
    parser.add_argument("--fine-tune", default=True)
    parser.add_argument("--set-start", default=False)
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--topK", default=TOPK)
    parser.add_argument("--T", default=T)
    parser.add_argument("--optim", default='sgd')
    parser.add_argument("--is-training", default=False,
                        help="Whether to freeze BN layers, False for Freezing")
    parser.add_argument("--img-dir", type=str, default=IMGSHOT_DIR,
                        help="Where to save images of the model.")
    parser.add_argument("--num-workers", default=16)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=5, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=200, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-jitter", default=True)
    parser.add_argument("--random-rotate", default=True)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-matnet-from", type=str, default=RESTORE_MATNET_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-segnet-from", type=str, default=RESTORE_SEGNET_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-spnet-from", default=RESTORE_SPNET_FROM)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--class-distri", default=CLASS_DISTRI)
    return parser.parse_args()


args = get_arguments()


def pr_poly(actual_step, max_iter, T):
    return 1 - ((float(actual_step*args.batch_size) / max_iter) ** T)*(base_dist)


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.CommonFeatureExtract)
    b.append(model.MatFeatureExtract)
    b.append(model.SegFeatureExtract)

    for i, layers in enumerate(b):
        for j in layers.parameters():
            if j.requires_grad:
                yield j


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.SegClassifier.parameters())

    for j in range(len(b)):
        for i in b[j]:
            if i.requires_grad:
                yield i


def loss_calc(pred, label, class_weight=None):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weight)).cuda()

    return criterion(pred, label)


def dist_loss_calc(pred, label, class_weight=None, alpha=1e-6):  # original:alpha, 1e-6
    b, c, h, w = pred.size()
    softmax = torch.nn.Softmax(dim=1)
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if class_weight is None:
        loss = -softmax(label)*logsoftmax(pred)
        loss = loss.sum(dim=(1, 2, 3)).mean()
    else:
        loss = -softmax(label)*logsoftmax(pred)
        class_weight = torch.from_numpy(class_weight).to(torch.float32)
        class_weight = class_weight.view(1, 5, 1, 1)
        class_weight = class_weight.repeat(b, 1, h, w).cuda()
        loss = loss*class_weight
        loss = loss.sum(dim=(1, 2, 3)).mean()
    return loss*alpha


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, actual_step):
    """Original Author: Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, actual_step * args.batch_size, MAX_ITER, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = 10 * lr


def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.float32)
    for i, t in enumerate(target.cpu().numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(new_target).long()


def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    h, w = map(int, args.input_size.split(','))

    cudnn.enabled = True
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    model = SemiSPNet(h=int(h / 8), w=int(w / 8),
                      K=args.topK,
                      num_classes=args.num_classes)
    model = model.cuda()

    if args.fine_tune:
        # load pretrained MatNet Model
        saved_matnet_state_dict = torch.load(args.restore_matnet_from)
        for name, _ in model.CommonFeatureExtract.state_dict().items():
            if "num_batches_tracked" in name.split('.'):
                continue
            ckpt_name = name.replace("model." + name.split('.')[1],
                                     "Scale." + resnet_feature_layers[int(name.split('.')[1])])
            model.CommonFeatureExtract.state_dict()[name].copy_(saved_matnet_state_dict[ckpt_name])

        for name, _ in model.MatFeatureExtract.state_dict().items():
            if "num_batches_tracked" in name.split('.'):
                continue
            ckpt_name = name.replace("model.", "Scale.layer4.")
            model.MatFeatureExtract.state_dict()[name].copy_(saved_matnet_state_dict[ckpt_name])

        # load pretrained SegNet Model
        saved_segnet_state_dict = torch.load(args.restore_segnet_from)
        for name, _ in model.SegFeatureExtract.state_dict().items():
            if "num_batches_tracked" in name.split('.'):
                continue
            ckpt_name = name.replace("model", "Scale.layer4")
            model.SegFeatureExtract.state_dict()[name].copy_(saved_segnet_state_dict[ckpt_name])
        print("==> finetuning from checkpoint '{}' and '{}'".format(args.restore_matnet_from,
                                                                    args.restore_segnet_from))

    elif args.restore_spnet_from:
        if os.path.isfile(args.restore_spnet_from):
            print("=> loading checkpoint '{}'".format(args.restore_spnet_from))
            checkpoint = torch.load(args.restore_spnet_from)

            if args.set_start:
                args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))
            model.load_state_dict(checkpoint["state_dict"])
            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.restore_spnet_from, args.start_step))
        else:
            raise NotImplementedError("=> no checkpoint found at '{}'".format(args.restore_spnet_from))
    else:
        raise NotImplementedError("NO CHECKPOINT LOADED!!!")

    print("optimizer:", args.optim)
    if not args.is_training:
        # Frozen BN
        # when training, the model will use the running means and the
        # running vars of the pretrained model.
        # But note that eval() doesn't turn off history tracking.
        for name, param in model.named_parameters():
            if name.find('bn') != -1:
                param.requires_grad = False
        print("Freezing BN layers")
        optimizer = optim.SGD([{'params': get_1x_lr_params(model), 'lr': args.learning_rate},
                               {'params': get_10x_lr_params(model), 'lr': 10 * args.learning_rate}],
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        model.eval()
    else:
        raise NotImplementedError("Normal BN layers")

    if args.set_start:
        adjust_learning_rate(optimizer, args.start_step)
        print("now learning rate is:", optimizer.param_groups[0]['lr'])

    model.cuda()

    cudnn.benchmark = True

    # todo rewriting your own dataloader
    trainloader = data.DataLoader(SegTrainSemi(data_dir=data_dir,
                                               mirror=args.random_mirror,
                                               color_jitter=args.random_jitter,
                                               rotate=args.random_rotate,
                                               mean=IMG_MEAN,
                                               num_examples=NUM_EXAMPLES_PER_EPOCH),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    src_seg_losses = AverageMeter()
    tar_seg_losses = AverageMeter()
    src_mat_losses = AverageMeter()
    tar_mat_losses = AverageMeter()
    src_losses = AverageMeter()
    tar_losses = AverageMeter()

    src_img_seg_accuracy = AverageMeter()
    src_inf_seg_accuracy = AverageMeter()
    tar_img_seg_accuracy = AverageMeter()
    tar_inf_seg_accuracy = AverageMeter()

    src_img_mat_accuracy = AverageMeter()
    src_inf_mat_accuracy = AverageMeter()
    tar_img_mat_accuracy = AverageMeter()
    tar_inf_mat_accuracy = AverageMeter()

    src_img_seg_mat_accuracy = AverageMeter()
    src_inf_seg_mat_accuracy = AverageMeter()
    tar_img_seg_mat_accuracy = AverageMeter()
    tar_inf_seg_mat_accuracy = AverageMeter()

    # weight computation
    class_distri = np.array(args.class_distri)
    normalized_class_distri = class_distri / np.sum(class_distri)
    class_weight = 1 / normalized_class_distri
    print("class weight:", class_weight)

    fake_lbl = np.zeros((h, w), dtype=np.uint8)

    cnt = 0
    src_inf_seg_acc = 0
    tar_inf_seg_acc = 0
    src_inf_mat_acc = 0
    tar_inf_mat_acc = 0
    actual_step = args.start_step
    while actual_step < args.final_step:
        iter_end = timeit.default_timer()
        for i_iter, batch in enumerate(trainloader):
            actual_step = int(args.start_step + cnt)

            # random choose a dataloader
            # 0: unsupervised, 1: semi-supervised, 2: supervised
            p_2 = pr_poly(actual_step, MAX_ITER, T=args.T)
            p_0 = p_1 = (1-p_2)/2
            tr_type = np.random.choice([0, 1, 2], p=[p_0, p_1, p_2])

            data_time.update(timeit.default_timer() - iter_end)

            src_imgs, tar_imgs, src_lbls, tar_lbls, unlabel_src_imgs, unlabel_tar_imgs, \
            src_name, tar_name, unlabel_src_name, unlabel_tar_name = batch

            bs = src_imgs.size(0)
            src_imgs = src_imgs.cuda() if tr_type == 2 else unlabel_src_imgs.cuda()
            tar_imgs = tar_imgs.cuda() if tr_type != 0 else unlabel_tar_imgs.cuda()
            src_lbls = src_lbls.cuda() if tr_type == 2 else None
            tar_lbls = tar_lbls.cuda() if tr_type != 0 else None
            src_name = src_name if tr_type == 2 else unlabel_src_name
            tar_name = tar_name if tr_type != 0 else unlabel_tar_name

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, actual_step)

            # todo rewriting vl2ch function (single value ==> one-hot representation)
            src_lbls_resize = vl2ch(src_lbls).cuda() if tr_type == 2 else None
            tar_lbls_resize = vl2ch(tar_lbls).cuda() if tr_type != 0 else None

            src_img_seg_lbl, tar_img_seg_lbl, src_img_mat_lbl, tar_img_mat_lbl = model(src_imgs, tar_imgs,
                                                                                       src_lbls_resize,
                                                                                       tar_lbls_resize,
                                                                                       tr_type)
            src_img = src_imgs.data.cpu().numpy()[0]
            tar_img = tar_imgs.data.cpu().numpy()[0]
            del src_imgs, tar_imgs

            src_lbls_reshape = resize_target(src_lbls, src_img_mat_lbl.size(2)) if tr_type == 2 else None
            tar_lbls_reshape = resize_target(tar_lbls, tar_img_mat_lbl.size(2)) if tr_type != 0 else None

            src_seg_loss = loss_calc(src_img_seg_lbl, src_lbls_reshape, class_weight) if tr_type == 2 else None
            tar_seg_loss = loss_calc(tar_img_seg_lbl, tar_lbls_reshape, class_weight) if tr_type != 0 else None

            src_mat_loss = loss_calc(src_img_mat_lbl, src_lbls_reshape, class_weight) if tr_type == 2 else None
            tar_mat_loss = loss_calc(tar_img_mat_lbl, tar_lbls_reshape, class_weight) if tr_type != 0 else None

            # measure the difference of two distributions
            src_loss = dist_loss_calc(src_img_seg_lbl, src_img_mat_lbl, class_weight)
            tar_loss = dist_loss_calc(tar_img_seg_lbl, tar_img_mat_lbl, class_weight)

            if tr_type == 2:
                loss = (src_seg_loss + tar_seg_loss + src_mat_loss + tar_mat_loss + src_loss + tar_loss)
            if tr_type == 1:
                loss = (tar_seg_loss + tar_mat_loss + src_loss + tar_loss)
            if tr_type == 0:
                loss = (src_loss + tar_loss)

            losses.update(loss.item(), bs)
            if tr_type == 2:
                src_seg_losses.update(src_seg_loss.item(), bs)
                src_mat_losses.update(src_mat_loss.item(), bs)
            if tr_type != 0:
                tar_seg_losses.update(tar_seg_loss.item(), bs)
                tar_mat_losses.update(tar_mat_loss.item(), bs)
            src_losses.update(src_loss.item(), bs)
            tar_losses.update(tar_loss.item(), bs)

            if tr_type == 2:
                src_img_seg_acc = _pixel_accuracy(src_img_seg_lbl.data.cpu().numpy(),
                                                  src_lbls_reshape.data.cpu().numpy())
                src_inf_seg_acc = _infant_accuracy(src_img_seg_lbl.data.cpu().numpy(),
                                                   src_lbls_reshape.data.cpu().numpy())

            if tr_type != 0:
                tar_img_seg_acc = _pixel_accuracy(tar_img_seg_lbl.data.cpu().numpy(),
                                                  tar_lbls_reshape.data.cpu().numpy())
                tar_inf_seg_acc = _infant_accuracy(tar_img_seg_lbl.data.cpu().numpy(),
                                                   tar_lbls_reshape.data.cpu().numpy())

            if tr_type == 2:
                src_img_mat_acc = _pixel_accuracy(src_img_mat_lbl.data.cpu().numpy(),
                                                  src_lbls_reshape.data.cpu().numpy())
                src_inf_mat_acc = _infant_accuracy(src_img_mat_lbl.data.cpu().numpy(),
                                                   src_lbls_reshape.data.cpu().numpy())

            if tr_type != 0:
                tar_img_mat_acc = _pixel_accuracy(tar_img_mat_lbl.data.cpu().numpy(),
                                                  tar_lbls_reshape.data.cpu().numpy())
                tar_inf_mat_acc = _infant_accuracy(tar_img_mat_lbl.data.cpu().numpy(),
                                                   tar_lbls_reshape.data.cpu().numpy())

            # the accuracy between the seg and mat of source image
            src_img_seg_mat_acc = _dist_pixel_accuracy(src_img_seg_lbl.data.cpu().numpy(),
                                                       src_img_mat_lbl.data.cpu().numpy())
            src_inf_seg_mat_acc = _dist_infant_accuracy(src_img_seg_lbl.data.cpu().numpy(),
                                                        src_img_mat_lbl.data.cpu().numpy())

            tar_img_seg_mat_acc = _dist_pixel_accuracy(tar_img_seg_lbl.data.cpu().numpy(),
                                                       tar_img_mat_lbl.data.cpu().numpy())
            tar_inf_seg_mat_acc = _dist_infant_accuracy(tar_img_seg_lbl.data.cpu().numpy(),
                                                        tar_img_mat_lbl.data.cpu().numpy())

            if tr_type == 2:
                src_img_seg_accuracy.update(src_img_seg_acc, bs)
                src_inf_seg_accuracy.update(src_inf_seg_acc, bs)
            if tr_type != 0:
                tar_img_seg_accuracy.update(tar_img_seg_acc, bs)
                tar_inf_seg_accuracy.update(tar_inf_seg_acc, bs)

            if tr_type == 2:
                src_img_mat_accuracy.update(src_img_mat_acc, bs)
                src_inf_mat_accuracy.update(src_inf_mat_acc, bs)
            if tr_type != 0:
                tar_img_mat_accuracy.update(tar_img_mat_acc, bs)
                tar_inf_mat_accuracy.update(tar_inf_mat_acc, bs)

            src_img_seg_mat_accuracy.update(src_img_seg_mat_acc, bs)
            src_inf_seg_mat_accuracy.update(src_inf_seg_mat_acc, bs)
            tar_img_seg_mat_accuracy.update(tar_img_seg_mat_acc, bs)
            tar_inf_seg_mat_accuracy.update(tar_inf_seg_mat_acc, bs)

            avg_inf_seg_acc = (src_inf_seg_acc + tar_inf_seg_acc)/2
            avg_inf_mat_acc = (src_inf_mat_acc + tar_inf_mat_acc)/2
            avg_inf_seg_mat_acc = (src_inf_seg_mat_acc + tar_inf_seg_mat_acc)/2

            loss.backward()
            optimizer.step()

            batch_time.update(timeit.default_timer() - iter_end)

            if actual_step % args.print_freq == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.4f} ({loss.avg:.4f})\n'
                      'src_seg_loss {src_seg_loss.val:.4f} ({src_seg_loss.avg:.4f})\t'
                      'tar_seg_loss {tar_seg_loss.val:.4f} ({tar_seg_loss.avg:.4f})\n'
                      'src_mat_loss {src_mat_loss.val:.4f} ({src_mat_loss.avg:.4f})\t'
                      'tar_mat_loss {tar_mat_loss.val:.4f} ({tar_mat_loss.avg:.4f})\n'
                      'src_loss {src_loss.val:.11f} ({src_loss.avg:.11f}) '
                      'tar_loss {tar_loss.val:.11f} ({tar_loss.avg:.11f})\n'
                      'src_img_seg_acc {src_img_seg_acc.val:.3f} ({src_img_seg_acc.avg:.3f})\t'
                      'src_inf_seg_acc {src_inf_seg_acc.val:.3f} ({src_inf_seg_acc.avg:.3f})\n'
                      'tar_img_seg_acc {tar_img_seg_acc.val:.3f} ({tar_img_seg_acc.avg:.3f})\t'
                      'tar_inf_seg_acc {tar_inf_seg_acc.val:.3f} ({tar_inf_seg_acc.avg:.3f})\n'
                      'src_img_mat_acc {src_img_mat_acc.val:.3f} ({src_img_mat_acc.avg:.3f})\t'
                      'src_inf_mat_acc {src_inf_mat_acc.val:.3f} ({src_inf_mat_acc.avg:.3f})\n'
                      'tar_img_mat_acc {tar_img_mat_acc.val:.3f} ({tar_img_mat_acc.avg:.3f})\t'
                      'tar_inf_mat_acc {tar_inf_mat_acc.val:.3f} ({tar_inf_mat_acc.avg:.3f})\n'
                      'src_img_seg_mat_acc {src_img_seg_mat_acc.val:.3f} ({src_img_seg_mat_acc.avg:.3f}) '
                      'src_inf_seg_mat_acc {src_inf_seg_mat_acc.val:.3f} ({src_inf_seg_mat_acc.avg:.3f})\n'
                      'tar_img_seg_mat_acc {tar_img_seg_mat_acc.val:.3f} ({tar_img_seg_mat_acc.avg:.3f}) '
                      'tar_inf_seg_mat_acc {tar_inf_seg_mat_acc.val:.3f} ({tar_inf_seg_mat_acc.avg:.3f})\n'
                      'p2 {p2:.3f} tr_type {tr_type} seg_acc {seg_acc:.3f} mat_acc {mat_acc:.3f} seg_mat_acc {seg_mat_acc:.3f}\n'
                    .format(
                    cnt, actual_step, args.final_step,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    src_seg_loss=src_seg_losses,
                    tar_seg_loss=tar_seg_losses,
                    src_mat_loss=src_mat_losses,
                    tar_mat_loss=tar_mat_losses,
                    src_loss=src_losses,
                    tar_loss=tar_losses,
                    src_img_seg_acc=src_img_seg_accuracy,
                    src_inf_seg_acc=src_inf_seg_accuracy,
                    tar_img_seg_acc=tar_img_seg_accuracy,
                    tar_inf_seg_acc=tar_inf_seg_accuracy,
                    src_img_mat_acc=src_img_mat_accuracy,
                    src_inf_mat_acc=src_inf_mat_accuracy,
                    tar_img_mat_acc=tar_img_mat_accuracy,
                    tar_inf_mat_acc=tar_inf_mat_accuracy,
                    src_img_seg_mat_acc=src_img_seg_mat_accuracy,
                    src_inf_seg_mat_acc=src_inf_seg_mat_accuracy,
                    tar_img_seg_mat_acc=tar_img_seg_mat_accuracy,
                    tar_inf_seg_mat_acc=tar_inf_seg_mat_accuracy,
                    p2=p_2,
                    seg_acc=avg_inf_seg_acc,
                    mat_acc=avg_inf_mat_acc,
                    seg_mat_acc=avg_inf_seg_mat_acc,
                    tr_type=tr_type
                ))

            if actual_step % args.save_img_freq == 0:
                # src: img, GT, Seg, Mat
                # tar: img, GT, Seg, Mat
                msk_size = tar_img_seg_lbl.size(2)
                src_img = src_img.transpose(1, 2, 0)
                tar_img = tar_img.transpose(1, 2, 0)
                src_img = cv2.resize(src_img, (msk_size, msk_size), interpolation=cv2.INTER_NEAREST)
                tar_img = cv2.resize(tar_img, (msk_size, msk_size), interpolation=cv2.INTER_NEAREST)
                src_img += IMG_MEAN
                tar_img += IMG_MEAN
                src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
                tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)

                src_lbl = src_lbls.data.cpu().numpy()[0] if src_lbls is not None else fake_lbl
                tar_lbl = tar_lbls.data.cpu().numpy()[0] if tar_lbls is not None else fake_lbl
                # todo rewriting your vl2im function
                src_lbl = vl2im(src_lbl)
                tar_lbl = vl2im(tar_lbl)

                src_img_seg_pred = src_img_seg_lbl.data.cpu().numpy()[0].argmax(axis=0)
                src_img_seg_pred = vl2im(src_img_seg_pred)

                tar_img_seg_pred = tar_img_seg_lbl.data.cpu().numpy()[0].argmax(axis=0)
                tar_img_seg_pred = vl2im(tar_img_seg_pred)

                src_img_mat_pred = src_img_mat_lbl.data.cpu().numpy()[0].argmax(axis=0)
                src_img_mat_pred = vl2im(src_img_mat_pred)

                tar_img_mat_pred = tar_img_mat_lbl.data.cpu().numpy()[0].argmax(axis=0)
                tar_img_mat_pred = vl2im(tar_img_mat_pred)

                new_im = Image.new('RGB', (msk_size * 4, msk_size * 2))
                new_im.paste(Image.fromarray(src_img.astype('uint8'), 'RGB'), (0, 0))
                new_im.paste(Image.fromarray(src_lbl.astype('uint8'), 'RGB'), (msk_size, 0))
                new_im.paste(Image.fromarray(src_img_seg_pred.astype('uint8'), 'RGB'), (msk_size * 2, 0))
                new_im.paste(Image.fromarray(src_img_mat_pred.astype('uint8'), 'RGB'), (msk_size * 3, 0))

                new_im.paste(Image.fromarray(tar_img.astype('uint8'), 'RGB'), (0, msk_size))
                new_im.paste(Image.fromarray(tar_lbl.astype('uint8'), 'RGB'), (msk_size, msk_size))
                new_im.paste(Image.fromarray(tar_img_seg_pred.astype('uint8'), 'RGB'), (msk_size * 2, msk_size))
                new_im.paste(Image.fromarray(tar_img_mat_pred.astype('uint8'), 'RGB'), (msk_size * 3, msk_size))

                new_im_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                              + '_' + src_name[0][:-4] + '_to_' + tar_name[0][-8:]
                new_im_file = os.path.join(args.img_dir, new_im_name)
                new_im.save(new_im_file)

            if actual_step % args.save_pred_every == 0 and cnt != 0:
                print('taking snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'state_dict': model.state_dict()},
                           osp.join(args.snapshot_dir,
                                    'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))

            if actual_step >= args.final_step:
                break
            cnt += 1

    print('save the final model ...')
    torch.save({'example': actual_step * args.batch_size,
                'state_dict': model.state_dict()},
               osp.join(args.snapshot_dir,
                        'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds')


def _pixel_accuracy(pred, target):
    accuracy_sum = 0.0
    for i in range(0, pred.shape[0]):
        out = pred[i].argmax(axis=0)
        accuracy = np.sum(out == target[i], dtype=np.float32) / out.size
        accuracy_sum += accuracy
    return accuracy_sum / pred.shape[0]


def _infant_accuracy(pred, target):
    accuracy_sum = 0.0
    cnt = 0
    for i in range(0, pred.shape[0]):
        out = pred[i].argmax(axis=0)
        tar = target[i]
        # delete normal pixels
        out = out[tar != 0]
        tar = tar[tar != 0]
        if len(out) == 0 or len(tar) == 0:
            accuracy = 0.0
        else:
            accuracy = np.sum(out == tar, dtype=np.float32) / out.size
            accuracy_sum += accuracy
            cnt += 1
    if cnt == 0:
        return 0.0
    return accuracy_sum / cnt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _dist_infant_accuracy(pred, target):
    accuracy_sum = 0.0
    cnt = 0
    for i in range(0, pred.shape[0]):
        out = pred[i].argmax(axis=0)
        tar = target[i].argmax(axis=0)
        # delete normal pixels
        out = out[tar != 0]
        tar = tar[tar != 0]
        if len(out) == 0 or len(tar) == 0:
            accuracy = 0.0
        else:
            accuracy = np.sum(out == tar, dtype=np.float32) / out.size
            accuracy_sum += accuracy
            cnt += 1
    if cnt == 0:
        return 0.0
    return accuracy_sum / cnt


def _dist_pixel_accuracy(pred, target):
    accuracy_sum = 0.0
    for i in range(0, pred.shape[0]):
        out = pred[i].argmax(axis=0)
        tar = target[i].argmax(axis=0)
        accuracy = np.sum(out == tar, dtype=np.float32) / out.size
        accuracy_sum += accuracy
    return accuracy_sum / pred.shape[0]


if __name__ == '__main__':
    main()
