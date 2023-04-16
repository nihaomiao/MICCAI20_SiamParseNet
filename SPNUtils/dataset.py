import os
import numpy as np
import cv2
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as F
import random
from SegUtils.transforms import im2vl, resize, base64tonpy
import json_tricks as json


class SegTrainSemi(data.Dataset):
    def __init__(self, data_dir, mirror, color_jitter, rotate, mean, num_examples):
        super(SegTrainSemi, self).__init__()
        self.img_size = 256
        self.jitter_transform = transforms.Compose([
            transforms.ColorJitter(brightness=64. / 255, contrast=0.25, saturation=0.25, hue=0.04)
        ])
        self.data_dir = data_dir
        self.mean = mean
        self.is_mirror = mirror
        self.is_jitter = color_jitter
        self.is_rotate = rotate
        self.num_examples = num_examples
        train_label_json = os.path.join(data_dir, "train_label.json")
        with open(train_label_json, "r") as f:
            self.train_label_dict = json.load(f)
        train_unlabel_json = os.path.join(data_dir, "train_unlabel.json")
        with open(train_unlabel_json, "r") as f:
            self.train_unlabel_dict = json.load(f)
        self.train_video_list = sorted(list(self.train_label_dict.keys()))

    def __len__(self):
        return int(self.num_examples)

    def __getitem__(self, idx):
        # random select a video
        chosen_video_idx = random.randrange(0, len(self.train_video_list))
        chosen_vid_name = self.train_video_list[chosen_video_idx]
        # random select two labeled frames and two unlabeled frames
        num_frames = len(self.train_label_dict[chosen_vid_name])
        unlabel_num_frames = len(self.train_unlabel_dict[chosen_vid_name])

        [src_frame_idx, tar_frame_idx] = random.sample(list(range(num_frames)), 2)
        [unlabel_src_frame_idx, unlabel_tar_frame_idx] = random.sample(list(range(unlabel_num_frames)), 2)

        src_img_name = os.path.basename(self.train_label_dict[chosen_vid_name][src_frame_idx]["image"])
        tar_img_name = os.path.basename(self.train_label_dict[chosen_vid_name][tar_frame_idx]["image"])
        unlabel_src_img_name = os.path.basename(self.train_unlabel_dict[chosen_vid_name][unlabel_src_frame_idx]["image"])
        unlabel_tar_img_name = os.path.basename(self.train_unlabel_dict[chosen_vid_name][unlabel_tar_frame_idx]["image"])

        src_img_arr = cv2.imread(self.train_label_dict[chosen_vid_name][src_frame_idx]["image"], cv2.IMREAD_COLOR)  # BGR
        tar_img_arr = cv2.imread(self.train_label_dict[chosen_vid_name][tar_frame_idx]["image"], cv2.IMREAD_COLOR)  # BGR
        assert src_img_arr.shape[2] == 3

        unlabel_src_img_arr = cv2.imread(self.train_unlabel_dict[chosen_vid_name][unlabel_src_frame_idx]["image"], cv2.IMREAD_COLOR)
        unlabel_tar_img_arr = cv2.imread(self.train_unlabel_dict[chosen_vid_name][unlabel_tar_frame_idx]["image"], cv2.IMREAD_COLOR)
        assert unlabel_src_img_arr.shape[2] == 3

        src_img_arr = resize(src_img_arr, self.img_size, interpolation=cv2.INTER_LINEAR)
        tar_img_arr = resize(tar_img_arr, self.img_size, interpolation=cv2.INTER_LINEAR)

        unlabel_src_img_arr = resize(unlabel_src_img_arr, self.img_size, interpolation=cv2.INTER_LINEAR)
        unlabel_tar_img_arr = resize(unlabel_tar_img_arr, self.img_size, interpolation=cv2.INTER_LINEAR)

        src_lbl_arr = base64tonpy(self.train_label_dict[chosen_vid_name][src_frame_idx]["label"])
        src_lbl_arr = im2vl(src_lbl_arr)
        tar_lbl_arr = base64tonpy(self.train_label_dict[chosen_vid_name][tar_frame_idx]["label"])
        tar_lbl_arr = im2vl(tar_lbl_arr)

        src_lbl_arr = resize(src_lbl_arr, self.img_size, interpolation=cv2.INTER_NEAREST)
        tar_lbl_arr = resize(tar_lbl_arr, self.img_size, interpolation=cv2.INTER_NEAREST)

        src_img_arr = cv2.cvtColor(src_img_arr, cv2.COLOR_BGR2RGB)
        src_img = Image.fromarray(src_img_arr)
        tar_img_arr = cv2.cvtColor(tar_img_arr, cv2.COLOR_BGR2RGB)
        tar_img = Image.fromarray(tar_img_arr)

        unlabel_src_img_arr = cv2.cvtColor(unlabel_src_img_arr, cv2.COLOR_BGR2RGB)
        unlabel_src_img = Image.fromarray(unlabel_src_img_arr)
        unlabel_tar_img_arr = cv2.cvtColor(unlabel_tar_img_arr, cv2.COLOR_BGR2RGB)
        unlabel_tar_img = Image.fromarray(unlabel_tar_img_arr)

        src_lbl = Image.fromarray(src_lbl_arr)
        tar_lbl = Image.fromarray(tar_lbl_arr)

        if self.is_jitter:
            src_img = self.jitter_transform(src_img)
            tar_img = self.jitter_transform(tar_img)
            unlabel_src_img = self.jitter_transform(unlabel_src_img)
            unlabel_tar_img = self.jitter_transform(unlabel_tar_img)

        if self.is_mirror:
            if random.random() < 0.5:
                src_img = F.hflip(src_img)
                tar_img = F.hflip(tar_img)
                src_lbl = F.hflip(src_lbl)
                tar_lbl = F.hflip(tar_lbl)
                unlabel_src_img = F.hflip(unlabel_src_img)
                unlabel_tar_img = F.hflip(unlabel_tar_img)

        if self.is_rotate:
            angle = random.randint(0, 3)*90
            src_img = F.rotate(src_img, angle)
            tar_img = F.rotate(tar_img, angle)
            src_lbl = F.rotate(src_lbl, angle)
            tar_lbl = F.rotate(tar_lbl, angle)
            unlabel_src_img = F.rotate(unlabel_src_img, angle)
            unlabel_tar_img = F.rotate(unlabel_tar_img, angle)

        src_lbl_arr = np.asarray(src_lbl, dtype=np.uint8)
        tar_lbl_arr = np.asarray(tar_lbl, dtype=np.uint8)

        src_img_arr = cv2.cvtColor(np.asarray(src_img), cv2.COLOR_RGB2BGR)
        src_img_arr = np.asarray(src_img_arr, np.float32)
        tar_img_arr = cv2.cvtColor(np.asarray(tar_img), cv2.COLOR_RGB2BGR)
        tar_img_arr = np.asarray(tar_img_arr, np.float32)

        unlabel_src_img_arr = cv2.cvtColor(np.asarray(unlabel_src_img), cv2.COLOR_RGB2BGR)
        unlabel_src_img_arr = np.asarray(unlabel_src_img_arr, np.float32)
        unlabel_tar_img_arr = cv2.cvtColor(np.asarray(unlabel_tar_img), cv2.COLOR_RGB2BGR)
        unlabel_tar_img_arr = np.asarray(unlabel_tar_img_arr, np.float32)

        src_img_arr -= self.mean
        src_img_arr = src_img_arr.transpose((2, 0, 1))
        tar_img_arr -= self.mean
        tar_img_arr = tar_img_arr.transpose((2, 0, 1))
        unlabel_src_img_arr -= self.mean
        unlabel_src_img_arr = unlabel_src_img_arr.transpose((2, 0, 1))
        unlabel_tar_img_arr -= self.mean
        unlabel_tar_img_arr = unlabel_tar_img_arr.transpose((2, 0, 1))

        return src_img_arr.copy(), tar_img_arr.copy(), \
               src_lbl_arr.copy(), tar_lbl_arr.copy(), \
               unlabel_src_img_arr.copy(), unlabel_tar_img_arr, \
               src_img_name, tar_img_name, \
               unlabel_src_img_name, unlabel_tar_img_name


class SegTest(data.Dataset):
    def __init__(self, data_dir, mean):
        super(SegTest, self).__init__()
        self.img_size = 256
        self.data_dir = data_dir
        self.mean = mean
        test_label_json = os.path.join(data_dir, "test_label.json")
        with open(test_label_json, "r") as f:
            self.test_label_dict = json.load(f)
        self.files = []
        for video_name in self.test_label_dict.keys():
            self.files = self.files + self.test_label_dict[video_name]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        datafiles = self.files[idx]
        image = cv2.imread(datafiles["image"], cv2.IMREAD_COLOR)  # BGR
        image = resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        image = np.asarray(image, np.float32)
        image -= self.mean
        image = image.transpose((2, 0, 1))
        label = base64tonpy(datafiles["label"])
        label = im2vl(label)
        label = resize(label, self.img_size, interpolation=cv2.INTER_NEAREST)
        return image.copy(), label.copy(), os.path.basename(datafiles["image"])


if __name__ == "__main__":
    pass
