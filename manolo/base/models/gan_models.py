import torch
import torch.nn as nn

from ..utils.data_synth_utils import onehot

class discriminator_small(nn.Module):
    def __init__(self, input_embed=True):
        super(discriminator_small, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        # if input_embed:
        #     in_channels = 794
        # else:
        in_channels = 794
        self.dis = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels = None, input_embed=True):
        if input_embed:
            c = self.label_emb(labels)
        else:
            c = onehot(labels) 

        x = torch.cat([x, c], 1)
        x = self.dis(x)
        return x         
    

class generator_small(nn.Module):
    def __init__(self, z_dimension):
        super(generator_small, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.gen = nn.Sequential(
            nn.Linear(z_dimension+10, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 784),
            nn.Tanh()
        )
     
    def forward(self, x, labels = None, input_embed=True):
        if input_embed:
            c = self.label_emb(labels)
        else:
            c = onehot(labels) 

        x = torch.cat([x, c], 1)
        x = self.gen(x)
        return x         
    

class discriminator_large(nn.Module):
    def __init__(self, input_embed=True):
        super(discriminator_large, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        in_channels = 794

        self.dis = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x, labels = None, input_embed=True):
        if input_embed:
            c = self.label_emb(labels)
        else:
            c = onehot(labels) 

        x = torch.cat([x, c], 1)
        x = self.dis(x)
        return x         
    

class generator_large(nn.Module):
    def __init__(self, z_dimension):
        super(generator_large, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.gen = nn.Sequential(
            nn.Linear(z_dimension+10, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()            
        )
  
    def forward(self, x, labels = None, input_embed=True):
        if input_embed:
            c = self.label_emb(labels)
        else:
            c = onehot(labels) 

        x = torch.cat([x, c], 1)
        x = self.gen(x)
        return x       


class generator_large_bn(nn.Module):
    def __init__(self, z_dimension):
        super(generator_large_bn, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.gen = nn.Sequential(
            nn.Linear(z_dimension+10, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()            
        )
  
    def forward(self, x, labels = None, input_embed=True):
        if input_embed:
            c = self.label_emb(labels)
        else:
            c = onehot(labels) 

        x = torch.cat([x, c], 1)
        x = self.gen(x)
        return x     

class discriminator_large_bn(nn.Module):
    def __init__(self, input_embed=True):
        super(discriminator_large_bn, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        in_channels = 794
        dropout_rate = 0.0
        self.dis = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels = None, input_embed=True):
        if input_embed:
            c = self.label_emb(labels)
        else:
            c = onehot(labels) 

        x = torch.cat([x, c], 1)
        x = self.dis(x)
        return x         
    



class generator_dcgan(nn.Module):
    def __init__(self, nz, ncond, nc, ngf, input_embed=False):
        super(generator_dcgan, self).__init__()
        self.in_filter = 1
        self.linear = nn.Sequential(nn.Linear(nz + ncond, self.in_filter*self.in_filter*128), nn.LeakyReLU())
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(128, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 1, 1, 0, bias=False),

        )

    def forward(self, input, label, input_embed = False):

        c = onehot(label) 
        input = torch.concat([input , c], dim=1)
        input = self.linear(input) # 6,272        
        input = input.view(input.size(0), 128, self.in_filter, self.in_filter)
        return self.main(input)        

    

class discriminator_dcgan(nn.Module):
    def __init__(self, ncond, nz, nc, ndf, input_embed = False):
        super(discriminator_dcgan, self).__init__()
        self.nz = nz
        self.linear = nn.Sequential(nn.Linear(ncond, nz), nn.LeakyReLU())
        # self.linear = nn.Sequential(nn.Linear(ncond, 3*32*32), nn.LeakyReLU())
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.flat = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(nz + (ndf * 8 * 2 * 2), 1),
            nn.Sigmoid()
        )

        

    def forward(self, input, label, input_embed = False):
        
        output = self.main(input)
        output_fl = self.flat(output)

        c = onehot(label) 
        c = self.linear(c)
        x = torch.concat([output_fl , c], dim=1)


        return self.dense(x)
    


class generator_dcgan_001(nn.Module):
    def __init__(self, nz, ncond, nc, ngf, input_embed=False):
        super(generator_dcgan_001, self).__init__()
        self.in_filter = 7
        self.cond_embed = 128
        self.linear = nn.Sequential(nn.Linear(nz + ncond, self.in_filter*self.in_filter*self.cond_embed), nn.LeakyReLU())
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.cond_embed, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 10 x 10
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 13 x 13
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
        )

    def forward(self, input, label, input_embed = False):

        c = onehot(label) 
        input = torch.concat([input , c], dim=1)
        input = self.linear(input) # 6,272        
        input = input.view(input.size(0), self.cond_embed, self.in_filter, self.in_filter)
        # embed()
        return self.main(input)        

    

class discriminator_dcgan_001(nn.Module):
    def __init__(self, ncond, nz, nc, ndf, input_embed = False):
        super(discriminator_dcgan_001, self).__init__()
        self.nz = nz
        self.linear = nn.Sequential(nn.Linear(ncond, nz), nn.LeakyReLU())
        # self.linear = nn.Sequential(nn.Linear(ncond, 3*32*32), nn.LeakyReLU())
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(nz + (ndf * 8 * 2 * 2), 1),
            nn.Sigmoid()
        )

        

    def forward(self, input, label, input_embed = False):
        
        output = self.main(input)
        output_fl = self.flat(output)

        c = onehot(label) 
        c = self.linear(c)
        x = torch.concat([output_fl , c], dim=1)


        return self.dense(x)
    



class generator_dcgan_002(nn.Module):
    def __init__(self, nz, ncond, nc, ngf, input_embed=False):
        super(generator_dcgan_002, self).__init__()
        self.in_filter = 7
        self.noise_embed = 128
        # self.label_embed = 50
        self.label_embed = 100
        self.nc = nc
        self.ncond = ncond
        self.label_emb = nn.Embedding(self.ncond, self.label_embed)
        self.cond_linear = nn.Sequential(nn.Linear(self.label_embed, 49), nn.LeakyReLU())        
        self.linear = nn.Sequential(nn.Linear(nz, self.in_filter*self.in_filter*self.noise_embed), nn.LeakyReLU())
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_embed + 1, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 10 x 10
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 13 x 13
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input, label, input_embed = False):
        input = self.linear(input) 
        input = input.view(input.size(0), self.noise_embed, self.in_filter, self.in_filter)
      
        c = self.label_emb(label)
        c = self.cond_linear(c)
        c = c.view(label.shape[0], 1, self.in_filter, self.in_filter)
        
        input = torch.concat([input , c], dim=1)

        return self.main(input)        

    

class discriminator_dcgan_002(nn.Module):
    def __init__(self, ncond, nz, nc, ndf, input_embed = False):
        super(discriminator_dcgan_002, self).__init__()
        self.nz = nz
        self.nc = nc
        # self.label_embed = 50
        self.label_embed = 100
        self.ncond = ncond
        self.label_emb = nn.Embedding(self.ncond, self.label_embed)
        self.cond_linear = nn.Sequential(nn.Linear(self.label_embed, 1*32*32), nn.LeakyReLU())
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        self.flat = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(ndf * 8 * 2 * 2, 1),
            nn.Sigmoid()
        )

        

    def forward(self, input, label, input_embed = False):
        c = self.label_emb(label)
        c = self.cond_linear(c)
        c = c.view(label.shape[0], 1, input.shape[2], input.shape[3])
        x = torch.concat([input , c], dim=1)

        output = self.main(x)
        output_fl = self.flat(output)

        return self.dense(output_fl)
    





