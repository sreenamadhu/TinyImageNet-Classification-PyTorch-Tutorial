# TinyImageNet-Classification-PyTorch-Tutorial

This repository provides PyTorch code to train a classification model on Tiny ImageNet dataset.

#### Requirements:

1. Download the Tiny ImageNet dataset from: http://cs231n.stanford.edu/tiny-imagenet-200.zip
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

2. Install PyTorch - https://pytorch.org


#### Code Structure:
1. dataset.py :: Has the code for Custom PyTorch Dataset class for Tiny ImageNet.
2. train.py :: Training code. Trains the model and saves the model for every epoch.
3. train_with_val.py :: Training code with model validation. For every epoch, we run train and validation and save the model only when the model improves on validation.
4. inference.py :: Loads the trained model and runs the model inference.
5. run.py :: Entrypoint code that loads and prepares the data, model and trains the model.
6. distributed_training.py :: Runs the distributed training in PyTorch using Data Parellelism - DistributedDataParallel
7. worker_analysis.py :: Analysis of num_workers parameter setting i.e., Time analysis of using subprocesses. 

#### Training:
To run model training, you can run the command: 
```
python run.py
```
Logs will be saved as logs.txt and Models will be saved in models/ folder. 

To run distributed model training on 1-5 GPUs, you can run the command: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run.py
```

To run model inference, you can run the command: 
```
python inference.py
```
