import torch 
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import torch.optim

from ..utils.data_synth_utils import onehot

class MNIST_cINN(nn.Module):
    '''cINN for class-conditional MNISt generation'''
    def __init__(self, lr, h, w):
        super().__init__()

        self.cond_size = 10
        self.h_im = h
        self.w_im = w
        self.ch = 1
        self.cinn = self.build_inn()


        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):

        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, ch_out))

        
        cond = Ff.ConditionNode(self.cond_size)
        nodes = [Ff.InputNode(self.ch, self.h_im, self.w_im)]

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, l, rev=False):
        z, jac = self.cinn(x, c=onehot(l), rev=rev)
        # jac = self.cinn.log_jacobian(run_forward=False)
        return z, jac

    def reverse_sample(self, z, l):
        return self.cinn(z, c=onehot(l), rev=True)


