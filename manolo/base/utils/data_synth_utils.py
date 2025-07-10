import argparse
import torch
import torch.nn as nn

from torchvision import transforms


def parser_function():
    parser = argparse.ArgumentParser(description='train base net')


    # net and dataset choosen
    parser.add_argument('--data_dir', type=str, default='mnist_data', help='name of folder that contains the dataset') 
    parser.add_argument('--model_name', type=str, default="cgan_large", help='name of the model')  
    parser.add_argument('--img_size', type=int, default=28, help='Input image size')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')

    # various path
    parser.add_argument('--save_root', type=str, default='./results', help='models visualizations, and logs are saved here')
    parser.add_argument('--output_path', type=str, default='./results', help='pSpecify the folder for the experiments')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-M', nargs='+', default = [], help='<Required> Set flag')

    # others
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--experiment_name', type=str, default='try', help='note for this run')

    # GANS
    parser.add_argument('--z_dimension', type=int, default=100, help='Size of the noise vector')
    parser.add_argument('--ngf', type=int, default=64, help='Number of filters in the generator')
    parser.add_argument('--ndf', type=int, default=64, help='Number of filters in the discriminator')

    # CINN
    parser.add_argument('--cinn_sigma_noise', type=float, default=0.08, help='initial learning rate')


    args, unparsed = parser.parse_known_args()

    return args, unparsed



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def initilise_data_synth_model(args, device):

    if args.model_name == "cgan_small":
        from ..models.gan_models import discriminator_small, generator_small
        D = discriminator_small()
        G = generator_small(args.z_dimension)

    elif args.model_name == "cgan_large":
        from ..models.gan_models import discriminator_large, generator_large
        D = discriminator_large()
        G = generator_large(args.z_dimension)        

    elif args.model_name == "cgan_large_bn":
        from ..models.gan_models import discriminator_large_bn, generator_large_bn
        D = discriminator_large_bn()
        G = generator_large_bn(args.z_dimension)

    elif args.model_name == "dcgan":
        from ..models.gan_models import discriminator_dcgan, generator_dcgan
        ch = 3
        D = discriminator_dcgan(args.num_classes, args.z_dimension, ch, args.ndf)
        G = generator_dcgan(args.z_dimension, args.num_classes, ch, args.ngf)  

    elif args.model_name == "dcgan_001":
        from ..models.gan_models import discriminator_dcgan_001, generator_dcgan_001
        ch = 3
        D = discriminator_dcgan_001(args.num_classes, args.z_dimension, ch, args.ndf)
        G = generator_dcgan_001(args.z_dimension, args.num_classes, ch, args.ngf)  

    elif args.model_name == "dcgan_002":
        from ..models.gan_models import discriminator_dcgan_002, generator_dcgan_002
        ch = 3
        D = discriminator_dcgan_002(args.num_classes, args.z_dimension, ch, args.ndf)
        G = generator_dcgan_002(args.z_dimension, args.num_classes, ch, args.ngf)        

    elif args.model_name == "MNIST_cINN":
        from ..models.cinn_models import MNIST_cINN
        D = None
        G = MNIST_cINN(args.lr, args.img_size, args.img_size)

    else:
        raise Exception('model name does not exist.')

    if D != None:
        D.apply(weights_init).to(device)
        G.apply(weights_init)

    return D, G.to(device)




def onehot(labels):
    num_img = len(labels)
    label_onehot = torch.zeros((num_img,10)).to(labels.device)
    label_onehot[torch.arange(num_img),labels]=1
    return label_onehot


## Data stuff
def unnormalize(x, data_mean, data_std):
    '''go from normaized data x back to the original range'''

    if type(data_mean) is tuple : 
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/data_std[0], 1/data_std[1], 1/data_std[2] ]),
                                transforms.Normalize(mean = [ -data_mean[0], -data_mean[1], -data_mean[2] ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        return invTrans(x)
    elif type(data_mean) is float :
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0.],
                                                     std = [ 1/data_std]),
                                transforms.Normalize(mean = [ -data_mean],
                                                     std = [ 1. ]),
                               ])
        return invTrans(x)    
    else:
        return x * data_std + data_mean

