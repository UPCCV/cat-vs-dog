# PyTorch 实践指南 

本文是文章[PyTorch实践指南](https://zhuanlan.zhihu.com/p/29024978)配套代码，请参照[知乎专栏原文](https://zhuanlan.zhihu.com/p/29024978)或者[对应的markdown文件](PyTorch实战指南.md)更好的了解而文件组织和代码细节, 本项目增加了配置文件，日志功能以及断点续训的能力.

## 数据下载
- 从[kaggle比赛官网](https://www.kaggle.com/c/dogs-vs-cats/data) 下载所需的数据；或者直接从此下载[训练集](http://pytorch-1252820389.file.myqcloud.com/data/dogcat/train.zip)和[测试集](http://pytorch-1252820389.file.myqcloud.com/data/dogcat/test1.zip)
- 解压并把训练集和测试集分别放在一个文件夹中

## 安装
- PyTorch : 可按照[PyTorch官网](http://pytorch.org)的指南，根据自己的平台安装指定的版本
- 安装指定依赖：

```
pip3 install -r requirements.txt
```

## 训练

```
./train.sh
```

## 测试

```
python3 test.py
```