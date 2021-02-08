---
title: ESPCN阅读笔记
date: 2018-03-27 22:38:27
tags: [Deep Learning, paper]
mathjax2: true
---

在超分辨中一个相对简单的改进，初次尝试看了很久，水平有待提高，这里简单记录一下这个idea



## 简介

论文的全称是《Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network》，既然要做到Real-Time，对处理速度就提出了一定的要求，网络结构和图像大小对速度有很大影响，这篇论文主要就是在图像大小上下功夫的

## 动机

超分辨是将输入从low-resolution（LR）重建成high-resolution（HR）输出，这篇论文之前的工作中，分辨率的提升主要发生在两个阶段：1. 神经网络中段 2. 第一层或第一层之前。这样先提升分辨率，然后在HR上进行后续操作会增加计算复杂度，典型的就是SRCNN（第一层就通过双三次插值先把图像提升到目标大小），能不能在减少计算量的同时保证超分辨的效果？

<!--more-->

## 思路

基于此，作者提出了一种网络架构：先用CNN从LR上提取特征，再用sub-pixel convolution层得到HR的目标

![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1522162719/ESPCN/ESPCN1.png>)

前两层就是普通的卷积，传统的插值函数的功能在这些卷积层中可以被学习到，更加灵活（作者分别尝试了relu和tanh作为激活函数，发现tanh在单张图片超分辨上的表现要优于relu），最后一层为作者提出的亚像素卷积层



> 先介绍一下亚像素卷积：
>
> 使用大小为$k_s$的滤波器，以stride $1/r$（r代表缩放比例，上文已经提到过）在第$L-1$层的output上进行卷积，卷积过程中，当卷积核中的一个权重对应多个像素时就不会被激活，也就是说，卷积核中包含的不同权重在卷积过程中会被周期性地激活



但是在论文中，作者是以另一种方式实现的，数学表达式写作：

![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1522163156/ESPCN/ESPCN2.png>)

由上式可以看出，本文的实现分为两步：

**Convolution**

使用卷积核 $W_L（n_{L-1} \times r^2*C \times k_l \times k_l）$（输入通道数，输出通道数，卷积核大小，卷积核大小）其中 $k_l = k_s / r$，且$k_l$为整数，对第$L-1$层的output上进行卷积，得到大小为 $H \times W \times C * {r^2}$的特征

可以看到，作者使用了$r$个大小为$k_l$的卷积核去代替原本大小为$k_s$的卷积和，这样步长就变为了整数，可以直接套用普通卷积进行计算，但是相应的，要对结果中的像素进行重新排列



**PS（periodic shuffling）**

从图上我们可以很容易看出，该操作的作用就是把$H \times W \times C * {r^2}$的数据用循环的方式（对应上面亚像素卷积的循环激活）排列成了$rH \times rW \times C$，输入中$1 \times 1 \times {r^2}$对应着输出中的$r \times r$，数学公式表达如下：

![](https://res.cloudinary.com/du3fbbzfy/image/upload/v1522163300/ESPCN/ESPCN3.png)

Loss function就是计算重建后图片与原始HR图片做pixel-wise的MSE

![]()

本文提出的模型相较于其他方法速度上有了显著的提升，重建效果也有所提高（具体比较见论文）

![]()

## 阅读体会

这篇论文想告诉我们的很重要的一点就是：**先提取特征，再进行其他操作能节省很多计算量**。类似的，R-CNN直接在原图上进行region proposal，Fast R-CNN先用卷积提取特征，在此基础上再使用region proposal，计算量主要集中在卷积操作上，在region proposal的过程中可以重复使用前面卷积层已经算好的结果，大大节省了总的计算量

## 参考

[《Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network》]( https://arxiv.org/abs/1609.05158)

[ 《Fast R-CNN》](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

