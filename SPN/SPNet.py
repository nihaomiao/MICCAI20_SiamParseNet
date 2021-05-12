import torch
import torch.nn as nn
import torchvision.models as models
from deeplab.model import Res_Deeplab
import torch.nn.functional as F
from ProUtils.misc import vl2ch
import random
random.seed(1234)

resnet_feature_layers = ['conv1',
                         'bn1',
                         'relu',
                         'maxpool',
                         'layer1',
                         'layer2',
                         'layer3',
                         'layer4',
                         'layer5']


class CommonFeatureExtraction(torch.nn.Module):
    def __init__(self, num_classes, train_cfe=True):
        super(CommonFeatureExtraction, self).__init__()
        self.model = Res_Deeplab(num_classes=num_classes)
        self.resnet_feature_layers = ['conv1',
                                      'bn1',
                                      'relu',
                                      'maxpool',
                                      'layer1',
                                      'layer2',
                                      'layer3']
        resnet_module_list = [self.model.conv1,
                              self.model.bn1,
                              self.model.relu,
                              self.model.maxpool,
                              self.model.layer1,
                              self.model.layer2,
                              self.model.layer3]
        self.model = nn.Sequential(*resnet_module_list)
        if not train_cfe:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image_batch):
        features = self.model(image_batch)
        return features


class MatFeatureExtraction(torch.nn.Module):
    def __init__(self, num_classes, train_nfe=True, normalization=True):
        super(MatFeatureExtraction, self).__init__()
        self.normalization = normalization
        self.model = Res_Deeplab(num_classes=num_classes).layer4
        if not train_nfe:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization:
            features = F.normalize(features, p=2, dim=1)
        return features


class SegFeatureExtraction(torch.nn.Module):
    def __init__(self, num_classes, train_sfe=True):
        super(SegFeatureExtraction, self).__init__()
        self.model = Res_Deeplab(num_classes=num_classes).layer4
        if not train_sfe:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image_batch):
        features = self.model(image_batch)
        return features


class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)  # nhm: fea_ij = (B_i, A_j)
        # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
        correlation_tensor = feature_mul.transpose(1, 2).view(b, h*w, h, w)
        return correlation_tensor


class KNNGeometric(torch.nn.Module):
    def __init__(self, h, w, num_classes, K=20, resize=True):
        super(KNNGeometric, self).__init__()
        self.K = K  # top K of KNN
        self.h = h
        self.w = w
        self.num_classes = num_classes
        gridH, gridW, _ = torch.meshgrid(torch.arange(self.h),
                                         torch.arange(self.w),
                                         torch.arange(self.K))
        self.gridH = gridH.permute(2, 0, 1)
        self.gridW = gridW.permute(2, 0, 1)

    def forward(self, correlation_tensor, src_lbl_batch_resize, resize=True):
        bs = correlation_tensor.size(0)
        h = correlation_tensor.size(2)
        w = correlation_tensor.size(3)
        correlation_tensor_v = correlation_tensor.reshape(bs, h, w, h, w)
        ids = correlation_tensor.argsort(dim=1, descending=True)[:, :self.K, :, :]
        ids_diff = torch.arange(bs * self.K) * (h * w)
        ids_diff = ids_diff.unsqueeze(0).repeat(h * w, 1).transpose(1, 0).flatten()
        new_ids = ids.flatten() + ids_diff.cuda()
        if resize:
            src_lbl_batch_resize = F.interpolate(src_lbl_batch_resize, (h, w))
        src_lbl_resize = src_lbl_batch_resize.unsqueeze(dim=2).repeat(1, 1, self.K, 1, 1)\
            .permute(1, 0, 2, 3, 4).contiguous().view(self.num_classes, -1)
        tar_lbl_resize = src_lbl_resize[:, new_ids].view(self.num_classes, bs, self.K, h, w)\
            .permute(1, 0, 2, 3, 4).contiguous()
        correlation_tensor_v = correlation_tensor_v.unsqueeze(dim=1)\
            .repeat(1, self.K, 1, 1, 1, 1).view(bs*self.K*h*w, h*w)
        new_gird = torch.arange(h * w)
        new_gird = new_gird.unsqueeze(0).repeat(bs * self.K, 1).flatten()
        tar_corr_resize = correlation_tensor_v[new_ids, new_gird].view(bs, self.K, h, w)
        tar_corr_resize = tar_corr_resize.unsqueeze(dim=1).repeat(1, self.num_classes, 1, 1, 1)
        tar_pred_resize = tar_lbl_resize * tar_corr_resize
        tar_pred_resize = tar_pred_resize.sum(dim=2)
        return tar_pred_resize


class SegFeatureClassification(torch.nn.Module):
    def __init__(self, num_classes, train_sfc=True):
        super(SegFeatureClassification, self).__init__()
        self.model = Res_Deeplab(num_classes=num_classes).layer5
        if not train_sfc:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image_batch):
        features = self.model(image_batch)
        return features


# Propagation Branch
class ProNet(torch.nn.Module):
    def __init__(self, h, w, K, num_classes=5, train_cfe=True, train_nfe=True):
        super(ProNet, self).__init__()
        self.CommonFeatureExtract = CommonFeatureExtraction(num_classes=num_classes, train_cfe=train_cfe)
        self.MatFeatureExtract = MatFeatureExtraction(num_classes=num_classes, train_nfe=train_nfe)
        self.FeatureCorrelation = FeatureCorrelation()
        self.KNNGeometric = KNNGeometric(h=h, w=w, K=K, num_classes=num_classes)

    def forward(self, src_img_batch, tar_img_batch, src_lbl_batch_resize):
        input_size = src_img_batch.size()[-1]
        src_img_fea_batch = self.CommonFeatureExtract(src_img_batch)
        src_img_fea_batch = self.MatFeatureExtract(src_img_fea_batch)
        tar_img_fea_batch = self.CommonFeatureExtract(tar_img_batch)
        tar_img_fea_batch = self.MatFeatureExtract(tar_img_fea_batch)

        correlation_tensor = self.FeatureCorrelation(src_img_fea_batch, tar_img_fea_batch)
        tar_prob_resize = self.KNNGeometric(correlation_tensor=correlation_tensor,
                                            src_lbl_batch_resize=src_lbl_batch_resize)
        return F.interpolate(tar_prob_resize, input_size, mode='bilinear')


# Segmentation Branch
class SegNet(torch.nn.Module):
    def __init__(self, num_classes=5, train_cfe=True, train_sfe=True, train_sfc=True):
        super(SegNet, self).__init__()
        self.CommonFeatureExtract = CommonFeatureExtraction(num_classes=num_classes, train_cfe=train_cfe)
        self.SegFeatureExtract = SegFeatureExtraction(num_classes=num_classes, train_sfe=train_sfe)
        self.SegClassifier = SegFeatureClassification(num_classes=num_classes, train_sfc=train_sfc)

    def forward(self, image_batch):
        input_size = image_batch.size()[2]
        image_batch = self.CommonFeatureExtract(image_batch)
        image_batch = self.SegFeatureExtract(image_batch)
        image_batch = self.SegClassifier(image_batch)
        return F.interpolate(image_batch, input_size, mode='bilinear')


class SPNet(torch.nn.Module):
    def __init__(self, h=32, w=32, K=20, num_classes=5, train_cfe=True, train_sfe=True, train_sfc=True, train_nfe=True):
        super(SPNet, self).__init__()
        # Shared Common Features
        self.CommonFeatureExtract = CommonFeatureExtraction(num_classes=num_classes, train_cfe=train_cfe)

        # Seg Branch
        self.SegFeatureExtract = SegFeatureExtraction(num_classes=num_classes, train_sfe=train_sfe)
        self.SegClassifier = SegFeatureClassification(num_classes=num_classes, train_sfc=train_sfc)

        # Prop Branch
        self.MatFeatureExtract = MatFeatureExtraction(num_classes=num_classes, train_nfe=train_nfe)
        self.FeatureCorrelation = FeatureCorrelation()
        self.KNNGeometric = KNNGeometric(h=h, w=w, K=K, num_classes=num_classes)

    def forward(self, src_img_batch, tar_img_batch, src_lbl_batch_resize, tar_lbl_batch_resize):
        input_size = src_img_batch.size()[-1]
        # common feature extraction
        src_img_com_fea = self.CommonFeatureExtract(src_img_batch)
        tar_img_com_fea = self.CommonFeatureExtract(tar_img_batch)

        # segment
        src_img_seg_fea = self.SegFeatureExtract(src_img_com_fea)
        tar_img_seg_fea = self.SegFeatureExtract(tar_img_com_fea)

        src_img_seg_lbl = self.SegClassifier(src_img_seg_fea)
        tar_img_seg_lbl = self.SegClassifier(tar_img_seg_fea)

        # propagation
        src_img_pro_fea = self.MatFeatureExtract(src_img_com_fea)
        tar_img_pro_fea = self.MatFeatureExtract(tar_img_com_fea)

        src_tar_correlation = self.FeatureCorrelation(src_img_pro_fea, tar_img_pro_fea)
        tar_src_correlation = self.FeatureCorrelation(tar_img_pro_fea, src_img_pro_fea)

        tar_img_pro_lbl = self.KNNGeometric(correlation_tensor=src_tar_correlation,
                                             src_lbl_batch_resize=src_lbl_batch_resize)
        src_img_pro_lbl = self.KNNGeometric(correlation_tensor=tar_src_correlation,
                                             src_lbl_batch_resize=tar_lbl_batch_resize)

        return F.interpolate(src_img_seg_lbl, input_size, mode='bilinear'), \
               F.interpolate(tar_img_seg_lbl, input_size, mode='bilinear'), \
               F.interpolate(src_img_pro_lbl, input_size, mode='bilinear'), \
               F.interpolate(tar_img_pro_lbl, input_size, mode='bilinear')


# semi-supervised learning
class SemiSPNet(torch.nn.Module):
    def __init__(self, h=32, w=32, K=20, num_classes=5, train_cfe=True, train_sfe=True, train_sfc=True, train_nfe=True):
        super(SemiSPNet, self).__init__()
        # Shared Common Features
        self.CommonFeatureExtract = CommonFeatureExtraction(num_classes=num_classes, train_cfe=train_cfe)

        # Segmentation Branch
        self.SegFeatureExtract = SegFeatureExtraction(num_classes=num_classes, train_sfe=train_sfe)
        self.SegClassifier = SegFeatureClassification(num_classes=num_classes, train_sfc=train_sfc)

        # Propagation Branch
        self.MatFeatureExtract = MatFeatureExtraction(num_classes=num_classes, train_nfe=train_nfe)
        self.FeatureCorrelation = FeatureCorrelation()
        self.KNNGeometric = KNNGeometric(h=h, w=w, K=K, num_classes=num_classes)

    def forward(self, src_img_batch, tar_img_batch,
                src_lbl_batch_resize, tar_lbl_batch_resize,
                tr_type):
        input_size = src_img_batch.size()[-1]
        # common feature extraction
        src_img_com_fea = self.CommonFeatureExtract(src_img_batch)
        tar_img_com_fea = self.CommonFeatureExtract(tar_img_batch)

        # segment
        src_img_seg_fea = self.SegFeatureExtract(src_img_com_fea)
        tar_img_seg_fea = self.SegFeatureExtract(tar_img_com_fea)

        src_img_seg_lbl = self.SegClassifier(src_img_seg_fea)
        tar_img_seg_lbl = self.SegClassifier(tar_img_seg_fea)

        # propagation
        src_img_pro_fea = self.MatFeatureExtract(src_img_com_fea)
        tar_img_pro_fea = self.MatFeatureExtract(tar_img_com_fea)

        src_tar_correlation = self.FeatureCorrelation(src_img_pro_fea, tar_img_pro_fea)
        tar_src_correlation = self.FeatureCorrelation(tar_img_pro_fea, src_img_pro_fea)

        if tr_type == 2:
            tar_img_pro_lbl = self.KNNGeometric(correlation_tensor=src_tar_correlation,
                                                src_lbl_batch_resize=src_lbl_batch_resize,
                                                resize=True)
        else:
            tar_img_pro_lbl = self.KNNGeometric(correlation_tensor=src_tar_correlation,
                                                src_lbl_batch_resize=src_img_seg_lbl,
                                                resize=False)
        if tr_type != 0:
            src_img_pro_lbl = self.KNNGeometric(correlation_tensor=tar_src_correlation,
                                                src_lbl_batch_resize=tar_lbl_batch_resize,
                                                resize=True)
        else:
            src_img_pro_lbl = self.KNNGeometric(correlation_tensor=tar_src_correlation,
                                                src_lbl_batch_resize=tar_img_seg_lbl,
                                                resize=False)

        return F.interpolate(src_img_seg_lbl, input_size, mode='bilinear'), \
               F.interpolate(tar_img_seg_lbl, input_size, mode='bilinear'), \
               F.interpolate(src_img_pro_lbl, input_size, mode='bilinear'), \
               F.interpolate(tar_img_pro_lbl, input_size, mode='bilinear')


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    src_img_batch = torch.rand((2, 3, 256, 256)).cuda()
    tar_img_batch = torch.rand((2, 3, 256, 256)).cuda()

    src_lbl_batch = torch.randint(high=5, size=(2, 256, 256)).cuda().to(dtype=torch.float32)
    tar_lbl_batch = torch.randint(high=5, size=(2, 256, 256)).cuda().to(dtype=torch.float32)

    src_lbl_batch_resize = vl2ch(src_lbl_batch).cuda()
    tar_lbl_batch_resize = vl2ch(tar_lbl_batch).cuda()

    model = SPNet()
    model = model.cuda()
    model.eval()

    src_img_seg_lbl, tar_img_seg_lbl, src_img_mat_lbl, tar_img_mat_lbl = model(src_img_batch,
                                                                               tar_img_batch,
                                                                               src_lbl_batch_resize,
                                                                               tar_lbl_batch_resize)
    print(src_img_seg_lbl.size(), tar_img_seg_lbl.size(),
          src_img_mat_lbl.size(), tar_img_mat_lbl.size())
