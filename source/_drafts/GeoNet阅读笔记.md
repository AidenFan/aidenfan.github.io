---
title: GeoNet阅读笔记
date: 2018-05-12 12:14:20
tags: [Deep Learning, paper]
---

GeoNet能用非监督、端对端的方式进行视频的深度、光流、运动估计，超越了之前非监督学习的方法，达到了和监督学习相仿的水平



由于视频方面数据集较少，难以包含所有可能的场景，所以无监督是非常有价值的，下面详细介绍一下这篇文章

## 看点

- 利用3D场景的几何性质——静态表面和动态目标的组合，用刚性结构重建器和非刚性运动定位器来分开学习刚性流和目标运动
- 以图像结构相似性（SSIM）来引导非监督学习
- 用几何一致性检验增强鲁棒性

<!--more-->

## GeoNet的结构

第一阶段 **rigid structure reconstructor**
- **DepthNet**负责深度（vgg/resnet50，几个连续帧作为一个batch输入，其中一帧作为reference，其他作为source）
- **PoseNet**负责相机移动（连续帧按色彩通道连起来作为输入，多次卷积得到相机的6个自由度，其实相当于在计算相机外参数）
- 两个网络分别回归再融合到一起 

第二阶段 **non-rigid motion localizer**
- **ResFlowNet**负责动态物体（resnet50学习残差非刚性流）

最后把第一阶段的rigid_flow和第二阶段的resflow合起来得到最终结果 

![]()

## Loss Function

非监督学习主要体现在loss的设置上，本文没有用到ground truth（如光流中将预测结果与ground truth之间的EPE来训练），这里主要参照了图像的结构相似性

由于网络是分阶段训练的，总的loss分别考虑了rigid structure reconstructor和non-rigid motion localizer的结果，下面是总loss的式子

![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1526099405/GeoNet/GeoNet2.png>)

可以看出，总loss由五部分组成

- **rigid_warp_loss**

![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1526099405/GeoNet/GeoNet3.png>)

计算target image和同一batch中其他source images warp后形成的图像的结构相似性，并结合了光流中的衡量标准

- **disp_smooth_loss**

![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1526099405/GeoNet/GeoNet4.png>)

对DepthNet预测的结果作了平滑处理

- **flow_warp_loss**
- **flow_smooth_loss**

这两个loss和上面两个loss相似，把rigid_flow换成full_flow，disp的smooth项换成full_flow的smooth项即可

- **flow_consistency_loss（几何一致性检验）** 

![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1526099405/GeoNet/GeoNet5.png>)

![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1526099405/GeoNet/GeoNet6.png>)是target image中一个像素p_t上full_flow的forward-backward consistency check

![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1526099405/GeoNet/GeoNet7.png>)当这个式子为True时，[...]就为1，反之为0（用来判断异常点）

> 举个栗子，image one上的点A经过forward对应到image two上的点B，点B又通过backward对应到image one上的点C，如果A和C相差很大，即违反了forward-backward consistency check，就可能是异常点，对这些异常点只计算他们的flow_smooth_loss



## 部分实现

由DepthNet得到的深度信息和PoseNet得到的相机外参数（6DOF）得到rigid_flow，实现利用了相机成像原理

```python
def compute_rigid_flow(depth, pose, intrinsics, reverse_pose=False):
  """Compute the rigid flow from target image plane to source image

  Args:
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source (or source to target if reverse_pose=True) 
          camera transformation matrix [batch, 6], in the order of 
          tx, ty, tz, rx, ry, rz; 
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Rigid flow from target image to source image [batch, height_t, width_t, 2]
  """
  batch, height, width = depth.get_shape().as_list()
  # Convert pose vector to matrix  
  # dimension:[batch, 6]-->[batch, 4, 4]
  pose = pose_vec2mat(pose)
  if reverse_pose:
    pose = tf.matrix_inverse(pose)
  # Construct pixel grid coordinates
  pixel_coords = meshgrid(batch, height, width)
  tgt_pixel_coords = tf.transpose(pixel_coords[:,:2,:,:], [0, 2, 3, 1])
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
  # Construct a 4x4 intrinsic matrix
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)
  # Get a 4x4 transformation matrix from 'target' camera frame to 'source' pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
  rigid_flow = src_pixel_coords - tgt_pixel_coords
  return rigid_flow
```



## Reference

《GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose》

[相机矩阵](https://blog.csdn.net/yangyong0717/article/details/73064823)