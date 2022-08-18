import torch


def vl2ch(img_tensor_batch):
    b, h, w = img_tensor_batch.size()
    img_tmp = torch.zeros(size=(b, 5, h, w), dtype=torch.float32)
    for ci in range(5):
        img_tmp[:, ci, :, :] = (img_tensor_batch.squeeze()==ci).to(dtype=torch.float32)
    return img_tmp


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
    label = label.cuda()
    if class_weight is None:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
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


if __name__ == '__main__':
    label_batch1_resize = torch.randint(high=5, size=(4, 1, 256, 256)).cuda().to(dtype=torch.float32)
    vl2ch(label_batch1_resize)
