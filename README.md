# RDSSM

双塔模型，评论生成系统的一个相似过滤模块。对生成的评论进行相似度计算，保留相似度高的作为candidates。



loss graph：

![](https://upload-images.jianshu.io/upload_images/1129359-117d7e20135cf1ed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



ps：

1.这边采取的是最简单的DNN的双塔，效果已经满足最低的需求了，没有在增加attention，lstm，conv等各种网络

2.这边没有采用原论文的1个正样本+4个负样本的方式，而是1对1的构造二分类，目的是满足query和doc是否相似这种场景的需求

3.这边建议走两个leakyrelu，这边实测单个leakyrelu的情况下loss下降慢，存在欠拟合的情况