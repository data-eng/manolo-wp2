from manolo.base.wrappers.other_packages import argparse
from manolo.base.wrappers.pytorch import torch_nn as nn
from manolo.base.wrappers.torchvision import torchvision

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
    parser.add_argument('--select_cuda', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=0)

    # others
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--exp_name', type=str, default='try', help='name for this run/experiment')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str, help='name of dataset', default='cifar10') # cifar10/cifar100
    parser.add_argument('--data_loader_type', type=str, default='custom_data_loader', help='manolo_data_tier, custom_data_loader, research_data_loader') # cifar10/cifar100
    parser.add_argument('--model_name', type=str, help='name of basenet', default="ResNet18")  # resnet20/resnet110

    # Debugging
    parser.add_argument('--use_wandb', type=str, default="False")


    parser.add_argument('--train_shuffle', type=str, default=False, help='shuffle training data') 
    parser.add_argument('--transform_train', type=str, default=False, help='Transform training data') 
    
    parser.add_argument('--train_feat_file', type=str, default="feat_out/train_feat_512_pretrained.pkl", help='output path to store training features') 
    parser.add_argument('--test_feat_file', type=str, default="feat_out/test_feat_512_pretrained.pkl", help='output path to store testing features')  

    parser.add_argument('--train_lab_file', type=str, default="feat_out/train_lab_512_pretrained.pkl", help='output path to store training labels') 
    parser.add_argument('--test_lab_file', type=str, default="feat_out/test_lab_512_pretrained.pkl", help='output path to store testing labels')  

    parser.add_argument('--pretrained_model', type=str, default='True', help='load pretrained weights') 

    parser.add_argument('--image_size', type=int, default=32, help='Adjust the input image to a given size (resize function)')

    # Evaluation stage:
    parser.add_argument('--input_fetures', type=int, default=512, help='Dimensionality of the input features (RN18 default 512D)')
    parser.add_argument('--epochs_eval_clf', type=int, default=40, help='Epochs to train the classifiers for the evaluation stage')
    parser.add_argument('--store_features_in_mlflow', type=str, default='True', help='Store extracted features in MLFlow.') 
    parser.add_argument('--store_metrics_in_mlflow', type=str, default='True', help='Store extracted features in MLFlow.') 


    args, unparsed = parser.parse_known_args()
    
    return args, unparsed



def initilise_architecture(args, device):

    ### WARNING: Still need to check if the following models are imported from the manolo torchvision
	
    if args.model_name == "ResNet18":
        # net = ResNet18(num_classes=args.num_classes).to(device)
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if args.pretrained_model=='True' else None 
        net = resnet18(weights=weights).to(device)
        feat_dim = net.fc.weight.shape[-1]
        net.fc = nn.Identity() # to get the 512-D features befre the lienar layer (classifier)

    elif args.model_name == "ResNet50":
        # net = ResNet18(num_classes=args.num_classes).to(device)
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V1 if args.pretrained_model=='True' else None 
        net = resnet50(weights=weights).to(device)
        feat_dim = net.fc.weight.shape[-1]
        net.fc = nn.Identity() # to get the 512-D features befre the lienar layer (classifier)


    elif args.model_name == "vgg11_bn":
        from torchvision.models import vgg11_bn, VGG11_BN_Weights

        weights = VGG11_BN_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg11_bn(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)  

    elif args.model_name == "vgg11":
        from torchvision.models import vgg11, VGG11_Weights

        weights = VGG11_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg11(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)  

    elif args.model_name == "vgg16_bn":
        from torchvision.models import vgg16_bn, VGG16_BN_Weights

        weights = VGG16_BN_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg16_bn(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)  

    elif args.model_name == "vgg16":
        from torchvision.models import vgg16, VGG16_Weights

        weights = VGG16_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg16(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)  

    elif args.model_name == "vgg19_bn":
        from torchvision.models import vgg19_bn, VGG19_BN_Weights

        weights = VGG19_BN_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg19_bn(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)   
		
    elif args.model_name == "vgg19":
        from torchvision.models import vgg19, VGG19_Weights

        weights = VGG19_Weights.DEFAULT if args.pretrained_model=='True' else None 
        net = vgg19(weights=weights).to(device)
        feat_dim = net.classifier[-1].weight.shape[-1]
        net.classifier[-1] = nn.Identity() # to get the features befre the lienar layer (classifier)        
    else:
        raise Exception('model name does not exist.')

    return net, feat_dim

