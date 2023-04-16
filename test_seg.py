# SPN w/o SSL
import sys
sys.path.append('/workspace/code/infant-project/SPN')  # todo the path to your code directory
import argparse
import numpy as np
import time
import torch
from torch.utils import data
from SPNUtils.dataset import SegTest
from SegUtils.transforms import CRF
import os
from SegUtils.transforms import vl2im, im2vl
from PIL import Image
from ProUtils.misc import Logger
from SPN.SPNet import SegNet
from SegUtils.metrics import mulDice, bpDice

# todo the image mean of dataset
IMG_MEAN = np.array((101.84807705937696, 112.10832843463207, 111.65973036298041), dtype=np.float32)
NUM_CLASSES = 5
BATCH_SIZE = 64
MAX_EPOCH = 20
TRAIN_BS = 20
postfix = "-spn-semi-anneal-T0.40-B0.67-seg"
TRAIN_EXAMPLE_PER_EPOCH = 1e4 if "semi" not in postfix else 3e4
TRAIN_STEP = MAX_EPOCH*TRAIN_EXAMPLE_PER_EPOCH//TRAIN_BS
INPUT_SIZE = (256, 256)
data_dir = '/data/youtube-infant-body-parsing'  # todo the path to testing data
root_dir = "/data/GMS-data/hfn5052/spn-youtube-infant"  # todo the path to save training results
GPU = "7"
# todo the path to your trained model
RESTORE_FROM = "/data/GMS-data/hfn5052/spn-youtube-infant/snapshots-spn-semi-anneal-T0.40-B0.67/B0020_S040000.pth"
res_dir = os.path.join(root_dir, "res"+postfix)
MSK_PATH = os.path.join(res_dir, "B%04dS%06d" % (TRAIN_BS, TRAIN_STEP), "msk")
os.makedirs(MSK_PATH, exist_ok=True)
CRF_PATH = os.path.join(res_dir, "B%04dS%06d" % (TRAIN_BS, TRAIN_STEP), "crf")
os.makedirs(CRF_PATH, exist_ok=True)
SHOW_PATH = os.path.join(res_dir, "B%04dS%06d" % (TRAIN_BS, TRAIN_STEP), "show")
os.makedirs(SHOW_PATH, exist_ok=True)
LOG_PATH = os.path.join(res_dir, "B%04dS%06d.log" % (TRAIN_BS, TRAIN_STEP))
sys.stdout = Logger(LOG_PATH, sys.stdout)
print("postfix:", postfix)
print("num of epoch:", MAX_EPOCH)
print(MSK_PATH)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SPN")
    parser.add_argument("--postfix", default=postfix)
    parser.add_argument("--num-epoch", type=int, default=MAX_EPOCH)
    parser.add_argument("--msk-path", default=MSK_PATH)
    parser.add_argument("--crf-path", default=CRF_PATH)
    parser.add_argument("--show-path", default=SHOW_PATH)
    parser.add_argument("--gpu",
                        help="choose gpu device.", default=GPU)
    parser.add_argument("--track-running-stats", default=True)  # set false to use current batch_stats when eval
    parser.add_argument("--momentum", default=0)  # set 0 to freeze running mean and var, useless when eval
    parser.add_argument("--data-dir", type=str, default=data_dir)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument('--print-freq', '-p', default=5, type=int,
                        metavar='N', help='print frequency')
    return parser.parse_args()


args = get_arguments()


def main():
    """Create the model and start the evaluation process."""
    print("Restored from:", args.restore_from)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = SegNet(args.num_classes)
    model.cuda()
    saved_state_dict = torch.load(args.restore_from)
    num_examples = saved_state_dict['example']
    print("num examples used in training:", num_examples)
    if args.track_running_stats:
        print("using running mean and running var")
        for name in model.state_dict():
            try:
                model.state_dict()[name].copy_(saved_state_dict['state_dict'][name])
            except KeyError:
                model.state_dict()[name].copy_(saved_state_dict['state_dict']["module." + name])
    else:
        print("using current batch stats instead of running mean and running var")
        raise NotImplementedError("if you froze BN when training, maybe you are wrong now!!!")

    model.eval()

    testloader = data.DataLoader(SegTest(data_dir, IMG_MEAN),
                                 batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    batch_time = AverageMeter()

    total_head_mIoU = []
    total_hand_mIoU = []
    total_body_mIoU = []
    total_foot_mIoU = []
    total_mdice = []

    total_head_mIoU_crf = []
    total_hand_mIoU_crf = []
    total_body_mIoU_crf = []
    total_foot_mIoU_crf = []
    total_mdice_crf = []

    with torch.no_grad():
        end = time.time()
        for index, (image, gts, name) in enumerate(testloader):
            output = model(image.cuda())
            Softmax = torch.nn.Softmax2d()
            softmax_output = Softmax(output)

            pred = torch.max(softmax_output, dim=1, keepdim=True)
            del output
            for ind in range(0, pred[0].size(0)):
                msk = torch.squeeze(pred[1][ind]).data.cpu().numpy()
                gt = gts[ind].data.cpu().numpy()
                # compute metric
                dice = mulDice(msk, gt)
                head_IoU = bpDice(msk, gt, 1)
                hand_IoU = bpDice(msk, gt, 2)
                body_IoU = bpDice(msk, gt, 3)
                foot_IoU = bpDice(msk, gt, 4)

                total_mdice.append(dice)
                total_head_mIoU.append(head_IoU)
                total_hand_mIoU.append(hand_IoU)
                total_body_mIoU.append(body_IoU)
                total_foot_mIoU.append(foot_IoU)

                msk = vl2im(msk)
                img_now = image[ind].data.cpu().numpy()
                img_now = img_now.transpose(1, 2, 0)
                img_now += IMG_MEAN

                # compute metric after using CRF
                msk_crf = CRF(img_now.astype('uint8').copy(), msk.astype('uint8').copy())
                dice_crf = mulDice(im2vl(msk_crf), gt)
                head_IoU_crf = bpDice(im2vl(msk_crf), gt, 1)
                hand_IoU_crf = bpDice(im2vl(msk_crf), gt, 2)
                body_IoU_crf = bpDice(im2vl(msk_crf), gt, 3)
                foot_IoU_crf = bpDice(im2vl(msk_crf), gt, 4)

                total_mdice_crf.append(dice_crf)
                total_head_mIoU_crf.append(head_IoU_crf)
                total_hand_mIoU_crf.append(hand_IoU_crf)
                total_body_mIoU_crf.append(body_IoU_crf)
                total_foot_mIoU_crf.append(foot_IoU_crf)

                msk = Image.fromarray(msk.astype('uint8'), 'RGB')
                mskfile = os.path.join(args.msk_path, name[ind][:-4]+".png")
                msk.save(mskfile)

                msk_crf = Image.fromarray(msk_crf.astype('uint8'), 'RGB')
                mskfile_crf = os.path.join(args.crf_path, name[ind][:-4]+".png")
                msk_crf.save(mskfile_crf)

                img_now = img_now[:, :, ::-1]  # BGR to RGB
                img_with_heatmap = np.float32(img_now) * 0.7 + np.float32(msk) * 0.3
                img_with_heatmap[np.float32(msk).sum(axis=2)==0] = img_now[np.float32(msk).sum(axis=2)==0]
                img_with_heatmap = Image.fromarray(img_with_heatmap.astype('uint8'), 'RGB')
                showfile = os.path.join(args.show_path, name[ind][:-4]+".png")
                img_with_heatmap.save(showfile)

            batch_time.update(time.time() - end)
            end = time.time()

            if index % args.print_freq == 0:
                print('Test:[{0}/{1}]\t'
                      'Time {batch_time.val:.3f}({batch_time.avg:.3f})'
                      .format(index, len(testloader), batch_time=batch_time)
                      )
    print(args.restore_from)
    print('head:{:.4f}'.format(np.array(total_head_mIoU).mean()),
          'hand:{:.4f}'.format(np.array(total_hand_mIoU).mean()),
          'body:{:.4f}'.format(np.array(total_body_mIoU).mean()),
          'foot:{:.4f}'.format(np.array(total_foot_mIoU).mean()),
          'dice:{:.4f}'.format(np.array(total_mdice).mean()))
    print('CRF head:{:.4f}'.format(np.array(total_head_mIoU_crf).mean()),
          'hand:{:.4f}'.format(np.array(total_hand_mIoU_crf).mean()),
          'body:{:.4f}'.format(np.array(total_body_mIoU_crf).mean()),
          'foot:{:.4f}'.format(np.array(total_foot_mIoU_crf).mean()),
          'dice:{:.4f}'.format(np.array(total_mdice_crf).mean()))


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


if __name__ == '__main__':
    main()


