import os
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision

from train_argument import parser, print_args

from time import time
from model import *
from utils import * 

from trainer import *

def main(args):
    save_folder = args.affix
    
    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, 'train', 'info')
    print_args(args, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic=True 
        torch.backends.cudnn.benchmark=False
    if args.model == "Lenet5":
        if args.mask:
            model = LeNet_5_Masked().to(device)
        else:
            model = LeNet_5().to(device)
    else:
        if args.mask:
            model = LeNet_300_100_Masked().to(device)
        else:
            model = LeNet_300_100().to(device)
    
    
    trainer = Trainer(args, logger)
   
    
    loss = nn.CrossEntropyLoss()

    tr_dataset = torchvision.datasets.MNIST(args.data_root, 
                                    train=True, 
                                    transform=torchvision.transforms.ToTensor(), 
                                    download=True)

    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # evaluation during training
    te_dataset = torchvision.datasets.MNIST(args.data_root, 
                                    train=False, 
                                    transform=torchvision.transforms.ToTensor(), 
                                    download=True)

    te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    trainer.train(model, loss, device, tr_loader, te_loader, optimizer=optimizer)
    


if __name__ == '__main__':
    args = parser()
    #print_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)