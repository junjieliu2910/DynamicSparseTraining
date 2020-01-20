import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from time import time


class Trainer():
    def __init__(self, args, logger, attack=None):
        self.args = args
        self.logger = logger
        self.attack = attack

    def train(self, model, loss, device, tr_loader, va_loader=None, optimizer=None, scheduler=None):
        args = self.args
        logger = self.logger

        _iter = 1

        begin_time = time()
        best_acc = 0.
        keep_ratio_at_best_acc = 0.
        best_keep_ratio = 1.
        acc_at_best_keep_ratio = 0.

        for epoch in range(1, args.max_epoch+1):
            logger.info("-"*30 + "Epoch start" + "-"*30)
            for data, label in tr_loader:
                data, label = data.to(device), label.to(device)        
                model.train()        
                output = model(data)
                loss_val = loss(output, label) 
                if args.mask:
                    for layer in model.modules():
                        if isinstance(layer, MaskedMLP) or isinstance(layer, MaskedConv2d):
                            loss_val += args.alpha * torch.sum(torch.exp(-layer.threshold))
                optimizer.zero_grad() 
                loss_val.backward()
                optimizer.step()

                if _iter % args.n_eval_step == 0:
                    logger.info('epoch: %d, iter: %d, spent %.2f s, training loss: %.3f' % (
                        epoch, _iter, time() - begin_time, loss_val.item()))

                    begin_time = time()
                
                _iter += 1
            cur_acc = self.test(model, device, va_loader)
            if args.mask:
                current_keep_ratio = print_layer_keep_ratio(model, logger)
            if cur_acc > best_acc:
                best_acc = cur_acc
                if args.mask:
                    keep_ratio_at_best_acc = current_keep_ratio
                filename = os.path.join(args.model_folder, 'best_acc_model.pth')
                save_model(model, filename)
            if args.mask and current_keep_ratio < best_keep_ratio:
                best_keep_ratio = current_keep_ratio
                acc_at_best_keep_ratio = cur_acc
                filename = os.path.join(args.model_folder, 'best_keepratio_model.pth')
                save_model(model, filename)
            if scheduler is not None:
                scheduler.step()
        logger.info(">>>>> Training process finish")
        if args.mask:
            logger.info("Best keep ratio {:.4f}, acc at best keep ratio {:.4f}".format(best_keep_ratio, acc_at_best_keep_ratio))
            logger.info("Best acc {:.4f}, keep ratio at best acc {:.4f}".format(best_acc, keep_ratio_at_best_acc))
        else:
            logger.info("Best test accuracy {:.4f}".format(best_acc))
        file_name = os.path.join(args.model_folder, 'final_model.pth')
        save_model(model, file_name)

    def test(self, model, device, loader):

        total_acc = 0.0
        num = 0
        model.eval()
        loss = nn.CrossEntropyLoss()
        std_loss = 0. 
        iteration = 0.
        with torch.no_grad():
            for data, label in loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum') 
                total_acc += te_acc
                num += output.shape[0]  
                std_loss += loss(output, label)
                iteration += 1
        std_acc = total_acc/num*100.
        std_loss /= iteration
        self.logger.info("Test accuracy {:.2f}%, Test loss {:.3f}".format(std_acc, std_loss))
        return std_acc

    
    