import torch


def vl2ch(img_tensor_batch):
    b, h, w = img_tensor_batch.size()
    img_tmp = torch.zeros(size=(b, 5, h, w), dtype=torch.float32)
    for ci in range(5):
        img_tmp[:, ci, :, :] = (img_tensor_batch.squeeze()==ci).to(dtype=torch.float32)
    return img_tmp


if __name__ == '__main__':
    label_batch1_resize = torch.randint(high=5, size=(4, 1, 256, 256)).cuda().to(dtype=torch.float32)
    vl2ch(label_batch1_resize)
