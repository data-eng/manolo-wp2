from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import shutil
import argparse
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
from IPython import embed

def parser_function():
    parser = argparse.ArgumentParser(description='train base net')

    # various path
    parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
    parser.add_argument('--img_root', type=str, default='./data', help='path name of image dataset')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=256, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-M','--milestones', action='append', help='<Required> Set flag', required=False)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=0)

    # others
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--note', type=str, default='try', help='note for this run')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str, help='name of dataset', default='cifar10') # cifar10/cifar100
    parser.add_argument('--model_architecture', type=str, help='name of basenet', default="ResNet18")  # resnet20/resnet110

    # Debugging
    parser.add_argument('--use_wandb', type=str, default="False")

    # MANOLO
    parser.add_argument('--loss_function', type=str, default="CE")
    parser.add_argument('--xai_scale', type=float, default=1.0, help='Scaling factor for the XAI term in the loss')
    parser.add_argument('--epsilon_xpression', type=float, default=1.0, help='Scaling factor for the Xpression metric')
	  
    
    parser.add_argument('--train_shuffle', type=str, default=False, help='shuffle training data') 
    parser.add_argument('--transform_train', type=str, default=False, help='shuffle training data') 
    
    parser.add_argument('--train_feat_file', type=str, default="feat_out/train_feat_512_pretrained.pkl", help='output path to store training features') 
    parser.add_argument('--test_feat_file', type=str, default="feat_out/test_feat_512_pretrained.pkl", help='output path to store testing features')  

    parser.add_argument('--train_lab_file', type=str, default="feat_out/train_lab_512_pretrained.pkl", help='output path to store training labels') 
    parser.add_argument('--test_lab_file', type=str, default="feat_out/test_lab_512_pretrained.pkl", help='output path to store testing labels')  

    parser.add_argument('--pretrained_model', type=str, default=True, help='load pretrained weights') 

    parser.add_argument('--image_size', type=int, default=32, help='Adjust the input image to a given size (resize function)')

    # Evaluation stage:
    parser.add_argument('--input_fetures', type=int, default=512, help='Dimensionality of the input features (RN18 default 512D)')
    parser.add_argument('--epochs_eval_clf', type=int, default=40, help='Epochs to train the classifiers for the evaluation stage')



    args, unparsed = parser.parse_known_args()
    
    return args, unparsed



def initilise_architecture(args, device):
	
    if args.model_architecture == "ResNet18":
        # net = ResNet18(num_classes=args.num_classes).to(device)
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if args.pretrained_model=='True' else None 
        net = resnet18(weights=weights).to(device)
        feat_dim = net.fc.weight.shape[-1]
        net.fc = nn.Identity() # to get the 512-D features befre the lienar layer (classifier)

    elif args.model_architecture == "ResNet50":
        # net = ResNet18(num_classes=args.num_classes).to(device)
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V1 if args.pretrained_model=='True' else None 
        net = resnet50(weights=weights).to(device)
        feat_dim = net.fc.weight.shape[-1]
        net.fc = nn.Identity() # to get the 512-D features befre the lienar layer (classifier)


    elif args.model_architecture == "vgg11_bn":
        from torchvision.models import vgg11_bn, VGG11_BN_Weights

        weights = VGG11_BN_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg11_bn(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)  

    elif args.model_architecture == "vgg11":
        from torchvision.models import vgg11, VGG11_Weights

        weights = VGG11_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg11(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)  

    elif args.model_architecture == "vgg16_bn":
        from torchvision.models import vgg16_bn, VGG16_BN_Weights

        weights = VGG16_BN_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg16_bn(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)  

    elif args.model_architecture == "vgg16":
        from torchvision.models import vgg16, VGG16_Weights

        weights = VGG16_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg16(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)  

    elif args.model_architecture == "vgg19_bn":
        from torchvision.models import vgg19_bn, VGG19_BN_Weights

        weights = VGG19_BN_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg19_bn(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)   
		
    elif args.model_architecture == "vgg19":
        from torchvision.models import vgg19, VGG19_Weights

        weights = VGG19_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg19(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)        
    else:
        raise Exception('model name does not exist.')

    return net, feat_dim





def count_parameters_in_MB(model):
	return sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val   = 0
		self.avg   = 0
		self.sum   = 0
		self.count = 0

	def update(self, val, n=1):
		self.val   = val
		self.sum   += val * n
		self.count += n
		self.avg   = self.sum / self.count



def create_exp_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))


def load_pretrained_model(model, pretrained_dict):
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict) 
	# 3. load the new state dict
	model.load_state_dict(model_dict)


def transform_time(s):
	m, s = divmod(int(s), 60)
	h, m = divmod(m, 60)
	return h,m,s


def save_checkpoint(state, is_best, save_root):
	save_path = os.path.join(save_root, 'checkpoint.pth.tar')
	torch.save(state, save_path)
	if is_best:
		best_save_path = os.path.join(save_root, 'model_best.pth.tar')
		shutil.copyfile(save_path, best_save_path)


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred    = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def kNN_features(trainloader, testloader, K, sigma, device):
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = np.asarray(trainloader.dataset.features)
    norm_trainFeatures = F.normalize(torch.Tensor(trainFeatures), dim=1).T

    trainLabels = torch.LongTensor(trainloader.dataset.targets)

    C = trainLabels.max() + 1

    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        # retrieval_one_hot = torch.zeros(K, C).cuda()
        retrieval_one_hot = torch.zeros(K, C).to(device)
        for batch_idx, (features, targets) in enumerate(testloader):
            end = time.time()
            # targets = targets.cuda(async=True)
            targets = targets.to(device)
            batchSize = features.size(0)
            # features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            norm_features = F.normalize(features, dim=1)

            dist = torch.mm(norm_features, norm_trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)

            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi).to(device)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_().to(device)
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                  total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))

    print("KNN accuracy: \t\t {} %".format(top1*100./total))

    return top1*100/total