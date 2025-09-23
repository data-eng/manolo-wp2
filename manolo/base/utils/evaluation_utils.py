import time
from manolo.base.wrappers.numpy import np
from manolo.base.wrappers.pytorch import torch
from manolo.base.wrappers.pytorch import nn_functional as F

def count_parameters_in_MB(model, req_grad=False):
    """Counts trainable parameters in the model in MB."""
    if req_grad:
        return sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

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