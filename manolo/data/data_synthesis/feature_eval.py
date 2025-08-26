from __future__ import absolute_import, print_function, division
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from manolo.base.data.dataset_utils import get_features_dataset

from manolo.base.utils.feat_utils import parser_function, AverageMeter, accuracy, kNN_features


def eval_features(args=None):
    """
    Run evaluation of features using both linear and non-linear classifiers.
    
    If `args` is None, command-line arguments will be parsed via `parser_function()`.
    
    Returns:
        results (dict): A dictionary with best accuracies and epochs for both classifiers and the kNN evaluation accuracy.
    """
    # If no args were provided, parse them.
    if args is None:
        args, unparsed = parser_function()
    
    # Set the save directory
    args.save_root = os.path.join(args.save_root, args.note)
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device configuration
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print("args = %s" % args)
    print("====== ====== ====== ====== ====== ====== ")    
    print("Using device:", device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')    
    print("====== ====== ====== ====== ====== ====== ")        

    # Get training and testing datasets (assumed to be based on precomputed features)
    train_loader, test_loader = get_features_dataset(args)
    
    # Define classifier architectures
    class ClfBase(nn.Module):
        def __init__(self, feat_in=512, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(feat_in, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        
    class LinearClf(nn.Module):
        def __init__(self, feat_in=512, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(feat_in, num_classes)

        def forward(self, x):
            return self.fc1(x)

    # Initialize the two classifiers and their optimizers
    clf_non_lin = ClfBase(args.input_fetures, args.num_classes).to(device)
    optimizer_non_lin = torch.optim.SGD(clf_non_lin.parameters(),
                                        lr=args.lr, 
                                        momentum=args.momentum, 
                                        weight_decay=args.weight_decay,
                                        nesterov=True)

    clf_lin = LinearClf(args.input_fetures, args.num_classes).to(device)
    optimizer_lin = torch.optim.SGD(clf_lin.parameters(),
                                    lr=args.lr, 
                                    momentum=args.momentum, 
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    args.milestones = [15, 25, 35]
    scheduler_non_lin = torch.optim.lr_scheduler.MultiStepLR(optimizer_non_lin, milestones=args.milestones, gamma=0.1)
    scheduler_lin = torch.optim.lr_scheduler.MultiStepLR(optimizer_lin, milestones=args.milestones, gamma=0.1)
    
    nb_epochs = args.epochs_eval_clf

    # kNN evaluation on features
    K = 200
    sigma = 0.1
    acc_knn = kNN_features(train_loader, test_loader, K, sigma, device)

    best_acc_non_lin = 0
    best_acc_lin = 0
    best_ep_non_lin = 0
    best_ep_lin = 0

    # Training loop
    for epoch in range(nb_epochs):
        # Train and then test the non-linear classifier
        _acc_non_lin, _ = train(train_loader, clf_non_lin, optimizer_non_lin, criterion,
                                 epoch, device, args.print_freq, clsf_type='\t Non linear')
        acc_non_lin, _ = test(test_loader, clf_non_lin, criterion, device)

        # Train and then test the linear classifier
        _acc_lin, _ = train(train_loader, clf_lin, optimizer_lin, criterion,
                              epoch, device, args.print_freq, clsf_type='Linear')
        acc_lin, _ = test(test_loader, clf_lin, criterion, device)     

        best_acc_non_lin, best_ep_non_lin = update_best_values(best_acc_non_lin, best_ep_non_lin, acc_non_lin, epoch)
        best_acc_lin, best_ep_lin = update_best_values(best_acc_lin, best_ep_lin, acc_lin, epoch)

        scheduler_non_lin.step()
        scheduler_lin.step()

    # Save the evaluation results to a file
    results_path = args.test_feat_file.split('.')[0] + '_evaluation_results_.txt'
    with open(results_path, 'w') as f:
        f.write("\nEvaluation of {}\n\n".format(args.test_feat_file.split('.')[0].split('/')[1]))
        f.write("Acc linear evaluation (best in ep {}): \t\t{}\n".format(best_ep_lin, best_acc_lin))
        f.write("Acc non linear evaluation (best in ep {}): \t{}\n".format(best_ep_non_lin, best_acc_non_lin))
        f.write("Acc KNN evaluation: \t\t\t\t\t\t{}\n".format(acc_knn))
    
    # Return the results as a dictionary
    results = {
        'best_acc_lin': best_acc_lin,
        'best_ep_lin': best_ep_lin,
        'best_acc_non_lin': best_acc_non_lin,
        'best_ep_non_lin': best_ep_non_lin,
        'acc_knn': acc_knn
    }
    return results


def update_best_values(best_value, best_epoch, current_value, current_epoch):
    if best_value < current_value:
        best_value = current_value
        best_epoch = current_epoch
    return best_value, best_epoch


def train(train_loader, net, optimizer, criterion, epoch, device, print_freq, clsf_type='Linear'):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    net.train()
    end = time.time()
    for i, (img, target) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        img = img.to(device)
        target = target.to(device)

        out = net(img)
        loss = criterion(out, target)

        prec1, prec5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            log_str = (
                'Epoch[{0}] - {clsf_type} :[{1:03}/{2:03}] '
                'Time:{batch_time.val:.4f} '
                'Data:{data_time.val:.4f}  '
                'loss:{losses.val:.4f}({losses.avg:.4f})  '
                'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                'prec@5:{top5.val:.2f}({top5.avg:.2f})'
            ).format(epoch, i, len(train_loader),
                     batch_time=batch_time, data_time=data_time,
                     losses=losses, top1=top1, top5=top5, clsf_type=clsf_type)
            print(log_str)
    return top1.avg, losses.avg


def test(test_loader, net, criterion, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    for i, (img, target) in enumerate(test_loader, start=1):
        img = img.to(device)
        target = target.to(device)
      
        with torch.no_grad():
            out = net(img)
            loss = criterion(out, target)

        prec1, prec5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    print('Evaluation metrics ==> Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(losses.avg, top1.avg, top5.avg))
    return top1.avg, top5.avg

