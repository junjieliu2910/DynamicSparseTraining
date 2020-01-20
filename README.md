# Installation 

Recommend to use python 3.7 and pytorch 1.2






# Mnist 

## LeNet300_100

```
cd mnist/
```

**Dense baseline**
```
python train.py --model=Lenet300_100 --affix=Lenet300_100_baseline
```

**Dynamic sparse training**
```
python3 train.py --model=Lenet300_100 --mask --alpha=0.0005 --affix=Lenet300_100_mask
```

## Lenet5_Caffe

```
cd mnist/
```

**Dense baseline**
```
python train.py --model=Lenet5 --affix=Lenet5_baseline
```

**Dynamic sparse training**
```
python train.py --model=Lenet5 --mask --alpha=0.0005 --affix=Lenet5_mask
```


## LSTM

### Set hyperparameter

You can set the corresponding hyperparameter in mnist_lstm/train.py

```
cd mnist_lstm/
```

**Dense baseline**


```
python train.py 
```

**Dynamic sparse training**

```
python train.py --mask
```

# Cifar10


## VGG16 

**Dense baseline**
```
python train.py --model=VGG16 --affix=VGG16_baseline
```

**Dynamic sparse training**
```
python train.py --model=VGG16 --mask --alpha=5e-6 --affix=VGG16_alpha5e-6
```

## WideResNet 

### Depth and widen factor
You need to change the depth and widen factor manually in cifar/train.py


**Dense baseline**
```
python train.py --model=WideResNet --affix=WideResNet_baseline
```

**Dynamic sparse training**
```
python train.py --model=WideResNet --mask --alpha=5e-6 --affix=WideResNet_masked
```


