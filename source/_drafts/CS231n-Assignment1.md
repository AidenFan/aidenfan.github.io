---
title: 'CS231n Assignment1'
date: 2018-03-23 22:10:45
tags: Deep Learning
---

最近在看CS231n的视频，顺便做了一下课后作业，感觉对理解算法和熟悉编程都是一个很好的训练，这里先总结一下Assignment1的作业



Assignment1的作业主要包括Image Classification, kNN, SVM, Softmax, Neural Network，用到的一些编程技巧对于新手非常实用，算法涉及到的相关公式、paper在CS231n给出的一个官方文档上都有给出（见文末链接）

对于Assignment1我的做法是首先理解公式，然后推出梯度，之后是代码层面的优化



## 关于数据集的一些问题

Windows环境下运行Tutorial中用来下载数据集的脚本get_datasets.sh可能会报错，这时候需要手动下载，在Assignment1中只用到了CIFAR-10，只要下载以后放在get_datasets.sh同目录下，再运行get_datasets.sh即可

（下载地址：https://www.cs.toronto.edu/~kriz/cifar.html）

<!--more-->

## kNN

用最近的k个点来确定带分类点的类别，计算复杂度较高，这部分作业的重点是对用numpy包来优化运算速度，这里给出一个简单的比较

```python
# two loops :
        dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
# one loop :
        dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis=1))
# no loop （完全平方公式）:
        dists = np.sqrt((-2 * np.dot(X, self.X_train.T)) + np.sum(X ** 2, axis=1, keepdims=True) + np.sum(self.X_train ** 2, axis=1))
```

我跑出来的速度如下，可以看到no loop的明显优势

> Two loop version took 57.234076 seconds
>
> One loop version took 111.937542 seconds
>
> No loop version took 0.621399 seconds



## SVM

`i`是第i条数据，`j`是第j类，`y_i`是第i条数据的正确类别

线性运算结果：![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1521820198/CS231n%20Assignment1/assignment1_1.png>)

Loss Function：![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1521820198/CS231n%20Assignment1/assignment1_2.png>)

Gradient：![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1521820198/CS231n%20Assignment1/assignment1_3.png>)



```python
def svm_loss(x, y):
    """
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    # correct_class_scores的大小(N,)
    # np.newaxis新增维度变为(N, 1)
    # 而x的大小是(N, C) 实际运算中应该使用了broadcast
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    # 正确项的loss
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    # 错误项的loss
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx
```

## Softmax

Loss Function：![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1521820198/CS231n%20Assignment1/assignment1_4.png>)

Gradient：（这个在Tutorial中没有，需要自己推一下，求个偏导）

![](<http://res.cloudinary.com/du3fbbzfy/image/upload/v1521820198/CS231n%20Assignment1/assignment1_5.png>)

```python
def softmax_loss(x, y):
    """
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    # 避免numerical problem
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    # 这一项待会算梯度的时候要用到 可以先单独算出来
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
```



## Hyperparameters tuning

Assignment1的最后一个部分是建立一个两层的简单神经网络，前向传播和BP在这就不细讲了，主要是应用链式法则求导

还有一个比较重要的问题也是作业每个部分都涉及的就是，超参数的设置，这对模型训练效果有非常大的影响，尤其是当神经网络变深时，对超参数的要求也比较高，Assignment1里给出的一个比较简单的解决方法是超参数搜索，这里以svm为例：

```python
# 只考虑了learning_rates和regularization_strengths
# 具体搜索区间可以自行设置
learning_rates = [1e-7, 5e-5]
regularization_strengths = [2.5e4, 5e4]

results = {}
best_val = -1  
best_svm = None

for l in learning_rates:
    for r in regularization_strengths:
        svm = LinearSVM()
        svm.train(X_train, y_train, learning_rate=l, reg=r, num_iters=1500)
        y_train_pred = svm.predict(X_train)
        y_val_pred = svm.predict(X_val)
        training_accuracy = np.mean(y_train == y_train_pred)
        validation_accuracy = np.mean(y_val == y_val_pred)
        results[(l, r)] = (training_accuracy, validation_accuracy)
        if validation_accuracy > best_val:
            best_val = validation_accuracy
            best_svm = svm
```

## Reference

CS231n官网笔记（对作业有很大帮助）：http://cs231n.github.io/

