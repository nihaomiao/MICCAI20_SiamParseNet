SiamParseNet
====

The code implementation of our MICCAI20 paper [SiamParseNet: Joint Body Parsing and Label Propagation in Infant Movement Videos
](https://arxiv.org/abs/2007.08646). 

(This reprository is still under construction.)

<div align=center><img src="SPN.png" width="585px" height="352px"/></div>

Quick Start
----
```python
import os
import torch
from SPN.SPNet import SPNet

# setting GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# setting x_m (src_img_batch) and x_n (tar_img_batch)
# size is (batch, channel, height, width)
src_img_batch = torch.rand((2, 3, 256, 256)).cuda()
tar_img_batch = torch.rand((2, 3, 256, 256)).cuda()

# setting y_m (src_lbl_batch) and y_n (tar_lbl_batch)
# size is (batch, height, width)
src_lbl_batch = torch.randint(high=5, size=(2, 256, 256)).cuda().to(dtype=torch.float32)
tar_lbl_batch = torch.randint(high=5, size=(2, 256, 256)).cuda().to(dtype=torch.float32)

# change to one-hot representation
# size is (batch, channel, height, width)
src_lbl_batch_resize = vl2ch(src_lbl_batch).cuda()
tar_lbl_batch_resize = vl2ch(tar_lbl_batch).cuda()

# using fully-supervised SPN for training
model = SPNet()
model = model.cuda()
model.eval()

# One forward during training
src_img_seg_lbl, tar_img_seg_lbl, src_img_mat_lbl, tar_img_mat_lbl = model(src_img_batch,
                                                                           tar_img_batch,
                                                                           src_lbl_batch_resize,
                                                                           tar_lbl_batch_resize)
print(src_img_seg_lbl.size(), tar_img_seg_lbl.size(),
      src_img_mat_lbl.size(), tar_img_mat_lbl.size())

```

DONE
----
The network structure of SiamParsetNet

TODO
----
The AAT training and MSI testing codes

Citing SiamParseNet
----
If you find our approaches useful in your research, please consider citing:
```
@inproceedings{ni2020siamparsenet,
  title={SiamParseNet: Joint Body Parsing and Label Propagation in Infant Movement Videos},
  author={Ni, Haomiao and Xue, Yuan and Zhang, Qian and Huang, Xiaolei},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={396--405},
  year={2020},
  organization={Springer}
}
```
For any problems with the code, please feel free to contact me: homerhm.ni@gmail.com

Acknowledgement
----
We acknowledge the code about DeepLab from [speedinghzl](https://github.com/speedinghzl/Pytorch-Deeplab).
