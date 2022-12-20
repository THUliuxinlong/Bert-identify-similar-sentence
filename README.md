# 智能信息第三次实验

> 学号：2022210953   姓名：刘鑫龙   班级：深数据硕221班
>

[TOC]

## 1、Explore Data Analyze

### 1. 关键词的数量

<img src="http://rmaux11hx.hn-bkt.clouddn.com/img/image-20221119110945705.png" alt="image-20221119110945705" style="zoom:70%;" />

### 2.不同标签的数量

<img src="http://rmaux11hx.hn-bkt.clouddn.com/img/image-20221119111025233.png" alt="image-20221119111025233" style="zoom:70%;" />

``` python
category  label
上呼吸道感染    0         958
          1         639
咳血        0         192
          1         128
哮喘        0         524
          1         350
感冒        0        1153
          1         768
支原体肺炎     0         471
          1         314
肺气肿       0         588
          1         392
肺炎        0         887
          1         588
胸膜炎       0         477
          1         318
label  label
0      0        5250
1      1        3497
```

### 3.统计句子的长度

![image-20221119111030981](http://rmaux11hx.hn-bkt.clouddn.com/img/image-20221119111030981.png)

## 2、模型训练

``` python
Epoch 10/10
----------
Train loss 0.03213179846299323 accuracy 0.9935978049617011
Val   loss 0.3737218607704909 accuracy 0.9330669330669331

test_acc 0.9330669330669331
              precision    recall  f1-score   support

  dissimilar   0.933931  0.955760  0.944719      1198
     similar   0.931701  0.899254  0.915190       804

    accuracy                       0.933067      2002
   macro avg   0.932816  0.927507  0.929955      2002
weighted avg   0.933036  0.933067  0.932860      2002
```

<img src="http://rmaux11hx.hn-bkt.clouddn.com/img/image-20221128162148523.png" alt="image-20221128162148523" style="zoom:80%;" />

<img src="http://rmaux11hx.hn-bkt.clouddn.com/img/image-20221128162200528.png" alt="image-20221128162200528" style="zoom:80%;" />

## 3、trick

### 1.数据增强

调用百度的api接口进行回译，query1有很多重复，只对query2做回译：

``` python
id,category,query1,query2,label
0,咳血,"剧烈运动后咯血,是怎么了?",剧烈运动后咯血是什么原因？,1
1,咳血,"剧烈运动后咯血,是怎么了?",剧烈运动后为什么会咯血？,1
2,咳血,"剧烈运动后咯血,是怎么了?",剧烈运动后咯血，应该怎么处理？,0
query2回译后：
0,咳血,"剧烈运动后咯血,是怎么了?",剧烈训练后咯血的原因是什么？,1
1,咳血,"剧烈运动后咯血,是怎么了?",为什么剧烈训练后会出现咯血？,1
2,咳血,"剧烈运动后咯血,是怎么了?",如何处理剧烈训练后的咯血？,0
```

```python
Epoch 10/10
----------
Train loss 0.024305461082955265 accuracy 0.9944530222450964
Val   loss 0.41488182583143784 accuracy 0.9295704295704296

test_acc 0.9295704295704296
              precision    recall  f1-score   support

  dissimilar   0.943001  0.939065  0.941029      1198
     similar   0.909765  0.915423  0.912585       804

    accuracy                       0.929570      2002
   macro avg   0.926383  0.927244  0.926807      2002
weighted avg   0.929653  0.929570  0.929606      2002
```

<img src="http://rmaux11hx.hn-bkt.clouddn.com/img/image-20221128162217700.png" alt="image-20221128162217700" style="zoom:80%;" />

<img src="http://rmaux11hx.hn-bkt.clouddn.com/img/image-20221128162238421.png" alt="image-20221128162238421" style="zoom:80%;" />

使用回译的方法，效果不明显。

### 2.对抗训练

使用PGD算法：

``` python
Epoch 10/10
----------
Train loss 0.03432506483122978 accuracy 0.9930261804047101
Val   loss 0.40538352633926217 accuracy 0.9315684315684316

test_acc 0.9315684315684316
              precision    recall  f1-score   support

  dissimilar   0.931652  0.955760  0.943552      1198
     similar   0.931436  0.895522  0.913126       804

    accuracy                       0.931568      2002
   macro avg   0.931544  0.925641  0.928339      2002
weighted avg   0.931565  0.931568  0.931333      2002
```

<img src="http://rmaux11hx.hn-bkt.clouddn.com/img/image-20221128193659795.png" alt="image-20221128193659795" style="zoom:80%;" />

<img src="http://rmaux11hx.hn-bkt.clouddn.com/img/image-20221128193708813.png" alt="image-20221128193708813" style="zoom:80%;" />

效果不明显。

### 3.模型集成

对RoBERTa-wwm-ext和BERT-wwm-ext做集成。

```python
              precision    recall  f1-score   support

  dissimilar   0.940545  0.950751  0.945621      1198
     similar   0.925411  0.910448  0.917868       804

    accuracy                       0.934565      2002
   macro avg   0.932978  0.930600  0.931744      2002
weighted avg   0.934467  0.934565  0.934475      2002
```

<img src="http://rmaux11hx.hn-bkt.clouddn.com/img/image-20221128202910379.png" alt="image-20221128202910379" style="zoom:80%;" />

集成后分类效果提升。
