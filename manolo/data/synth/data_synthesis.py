import os
import time
import random
from manolo.base.wrappers.numpy import np
from manolo.base.wrappers.pytorch import cudnn, torch
from manolo.base.wrappers.pytorch import nn_functional as F
from manolo.base.wrappers.pytorch import torch_nn as nn
from manolo.base.wrappers.torchvision import torchvision
from torchvision.utils import save_image
from torchvision import transforms

from manolo.base.metrics.Accuracy import accuracy
from manolo.base.data.data_loader import load_dataset
from manolo.base.utils.evaluation_utils import AverageMeter, kNN_features
from manolo.data.synth.feature_extraction_utils import parser_function
from manolo.base.metrics.code_carbon_utils import codecarbon_manolo
from manolo.base.utils.logger_utils import log_metrics

from manolo.base.data.data_loader import load_dataset_data_synth
from manolo.base.models.network_initializer import initialize_data_synth
from manolo.base.utils.data_synth_utils import unnormalize

from matplotlib import pyplot as plt


def img_tile(imgs, row_col = None, transpose = False, channel_first=True, channels=3):
    '''tile a list of images to a large grid.
    imgs:       iterable of images to use
    row_col:    None (automatic), or tuple of (#rows, #columns)
    transpose:  Wheter to stitch the list of images row-first or column-first
    channel_first: if true, assume images with CxWxH, else WxHxC
    channels:   3 or 1, number of color channels '''
    if row_col == None:
        sqrt = np.sqrt(len(imgs))
        rows = np.floor(sqrt)
        delt = sqrt - rows
        cols = np.ceil(rows + 2*delt + delt**2 / rows)
        rows, cols = int(rows), int(cols)
    else:
        rows, cols = row_col

    if channel_first:
        h, w = imgs[0].shape[1], imgs[0].shape[2]
    else:
        h, w = imgs[0].shape[0], imgs[0].shape[1]

    show_im = np.zeros((rows*h, cols*w, channels))

    if transpose:
        def iterator():
            for i in range(rows):
                for j in range(cols):
                    yield i, j

    else:
        def iterator():
            for j in range(cols):
                for i in range(rows):
                    yield i, j

    k = 0
    for i, j in iterator():

            im = imgs[k]
            if channel_first:
                im = np.transpose(im, (1, 2, 0))

            show_im[h*i:h*i+h, w*j:w*j+w] = im
            
            k += 1

            if k == len(imgs):
                break

    return np.squeeze(show_im)



def sample_outputs(args, sigma, device):
    '''Produce a random latent vector with sampling temperature sigma'''
    return sigma * torch.randn(args.batch_size, args.img_size*args.img_size).to(device)


def generate_from_group_seed(args, model, indexes_in, group_name, \
                             data_mean, data_std, c_test, x_test, device, \
                                label = None, exp_path=None, ref_img=False):
    '''....
    index_in:   Index of the validation image to use for the transfer.
    save_as:    Optional filename to save the image.'''

    samples_to_plot = 10
    temp = 0.000001
    n_dim = args.img_size * args.img_size

    z_sample_0 = sample_outputs(args, temp, device)
    z_0 = z_sample_0[0].expand_as(z_sample_0)[:samples_to_plot]

    interpolation_steps = np.linspace(0., 1., samples_to_plot, endpoint=True)
    t = torch.Tensor([list(interpolation_steps)]*n_dim).T.to(device)

    with torch.no_grad():

        l = c_test[indexes_in]
        z_reference_all, _jac = model(x_test[indexes_in], l)
        z_ref_centroid = z_reference_all.mean(dim=0)

        z_ref_centr = torch.stack([z_ref_centroid]*samples_to_plot)

        all_labs_imgs_generated = []
        for l_i in range(args.num_classes): 

            l = torch.LongTensor([l_i]*samples_to_plot).to(device)

            interp_z = (1.-t) * z_0 + t * z_ref_centr
            imgs_generated, _jac = model(interp_z, l, rev=True)

            imgs_generated = unnormalize(imgs_generated, data_mean, data_std)

            all_labs_imgs_generated.extend(imgs_generated.cpu())

    img_show = img_tile(all_labs_imgs_generated, (10,10), transpose=False, channel_first=True, channels=1)

    
    if ref_img:
        ref_img = img_tile(x_test[indexes_in].cpu(), (1,10), transpose=False, channel_first=True, channels=1)
        plt.imsave('{}/visualizing_reference_style_conditioned_{}.png'.format(exp_path, group_name), \
                    img_show, cmap='gray', vmin=0, vmax=1)        
       
    plt.imsave('{}/visualizing_style_conditioned_{}.png'.format(exp_path, group_name), \
                img_show, cmap='gray', vmin=0, vmax=1)


def generate_samples_style_conditioned(args, model_to_load, indexes_in, group_name, \
                                       exp_path='./results', num_class_cond = 10):

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == torch.device('cuda'):
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

    _D_model, model = initialize_data_synth(args, device)
    _, _, test_loader, data_mean, data_std = load_dataset_data_synth(args.data_dir, args.batch_size)

    model.to(device)

    # state_dict = {k:v for k,v in torch.load(model_to_load).items() if 'tmp_var' not in k}
    # embed()
    state_dict = torch.load(model_to_load)
    model.load_state_dict(state_dict)

    model.eval()

    x_test = []
    c_test = []
    for x,cc in test_loader:
        x_test.append(x)
        c_test.append(cc)
    x_test, c_test = torch.cat(x_test, dim=0).to(device), torch.cat(c_test, dim=0).to(device)

    generate_from_group_seed(args, model, indexes_in, group_name, \
                             data_mean, data_std, c_test, x_test, device, exp_path = exp_path)

    return 0



def show_samples(args, label, model, data_mean, data_std, ch, label_tags, exp_path, device):
    '''produces and shows samples for a given label (0-9)'''

    N_samples = 100
    l = torch.LongTensor(N_samples).to(device)
    l[:] = label

    
    z = torch.randn(N_samples, args.z_dimension).to(device)

    with torch.no_grad():
        if hasattr(model, 'cinn'):
            samples = model.reverse_sample(z, l)[0]
        else:
            samples = model(z, l)

    if ch == 1:
        with torch.no_grad():
            samples_unnorm = unnormalize(samples.view(z.shape[0],ch,args.img_size,args.img_size), data_mean, data_std)
            samples = samples_unnorm.cpu().numpy()    

    elif ch == 3: 
        with torch.no_grad():
            samples_unnorm = unnormalize(samples, data_mean, data_std)
            samples = samples_unnorm.cpu().numpy()
    
    save_image(samples_unnorm[:100], '{}/visualizing_eval_{}.png'.format(exp_path, label_tags[label]), nrow=10)      


def generate_samples_class_conditioned(args, model_to_load, exp_path, num_class_cond = 10):

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == torch.device('cuda'):
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True


    _, _, _, data_mean, data_std = load_dataset_data_synth(args.data_dir, args.batch_size)

    _D_model, model = initialize_data_synth(args, device)
    
    # embed()

    state_dict = torch.load(model_to_load)
    # state_dict = {k:v for k,v in torch.load(model_to_load).items() if 'tmp_var' not in k}
    model.load_state_dict(state_dict)

    model.eval()

    if "cifar" in args.data_dir:
        ch = 3
        cifar_10_classes = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
        data_tags = cifar_10_classes
    elif "fashion" in args.data_dir:
        ch = 1
        fashion_mnist_10_classes = {0:"T-shirttop", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}
        data_tags = fashion_mnist_10_classes
    elif "mnist" in args.data_dir:
        ch = 1
        mnist_10_classes = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9"}
        data_tags = mnist_10_classes

    for i in range(num_class_cond):
        show_samples(args, i, model, data_mean, data_std, ch, data_tags, exp_path, device)




def train_gan(args, exp_path):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == torch.device('cuda'):
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

    train_loader, _, _, data_mean, data_std = load_dataset_data_synth(args.data_dir, args.batch_size)

    # Initialize model
    D, G = initialize_data_synth(args, device)


    ##########################################################
    ######## Training ########
    criterion = nn.BCELoss() 
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

    t_start = time.time()

    print('Epoch\tBatch/Total \tTime \tD loss\tG loss')

    n_repeat_D = 1
    n_repeat_G = 1
    smoothGANlabels = False
    for epoch in range(args.epochs):

        for i, (img, label) in enumerate(train_loader):
            num_img = img.size(0)

            if img.shape[1] == 1:
                img = img.view(num_img,  -1)
            real_img = img.to(device)
            if not smoothGANlabels:
                real_label = torch.ones((num_img,1)).to(device)
                fake_label = torch.zeros((num_img,1)).to(device)
            else:
                real_label = torch.ones((num_img,1)).to(device) * 0.9
                fake_label = torch.ones((num_img,1)).to(device) * 0.1

            ##### Train the discriminator    
            for _ in range(n_repeat_D):
                l = label.to(device)
                z_l = torch.LongTensor(np.random.randint(0, args.num_classes, img.size(0))).to(device)
                z = torch.randn(num_img, args.z_dimension).to(device)

                real_out = D(real_img, l) 
                d_loss_real = criterion(real_out, real_label)
                
                fake_img = G(z, z_l)
                fake_out = D(fake_img, z_l)
                d_loss_fake = criterion(fake_out, fake_label)

                d_loss = d_loss_real + d_loss_fake 
                d_optimizer.zero_grad() 
                d_loss.backward() 
                d_optimizer.step() 

            ##### Train the generator
            for _ in range(n_repeat_G):

                z = torch.randn(num_img, args.z_dimension).to(device)

                fake_img = G(z, l)
                output = D(fake_img, l)
                g_loss = criterion(output, real_label)
                g_optimizer.zero_grad() 
                g_loss.backward()  
                g_optimizer.step()  


            if (i + 1) % args.print_freq == 0:
                print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f' % (epoch,
                                                                i, len(train_loader),
                                                                (time.time() - t_start)/60.,
                                                                d_loss.item(),
                                                                g_loss.item(),
                                                                ), flush=True)

            visualize_every_epoch = False
            if epoch == 0:
                if 'mnist' in args.data_dir:
                    real_images = real_img.cpu().clamp(0,1).view(-1,1,28,28).data
                    save_image(real_images, './{}/train_visualizing_real_images.png'.format(exp_path))
                else:
                    save_image(unnormalize(real_img, data_mean, data_std), './{}/real_images.png'.format(exp_path))
            if i == len(train_loader)-1 and visualize_every_epoch:
                if 'mnist' in args.data_dir:
                    fake_images = fake_img.cpu().clamp(0,1).view(-1,1,28,28).data
                    save_image(fake_images, './{}/train_visualizing_fake_images-{}.png'.format(exp_path, epoch + 1))
                else:
                    save_image(unnormalize(fake_img, data_mean, data_std), './{}/train_visualizing_fake_images-{}.png'.format(exp_path, epoch + 1))

        torch.save(G.state_dict(), './{}/generator.pth'.format(exp_path))
        torch.save(D.state_dict(), './{}/discriminator.pth'.format(exp_path))




def train_cinn(args, exp_path):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == torch.device('cuda'):
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

    train_loader, _, test_loader, _, _ = load_dataset_data_synth(args.data_dir, args.batch_size)

    # Initialize model
    _, cinn = initialize_data_synth(args, device)


    cinn.to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, milestones=args.M, gamma=0.1)

    # CINN-specific arrangement of training and validation data (this is to follow the literature, feel free to adopt a more standard arrangement)
    val_x, val_l = zip(*list(train_loader.dataset[i] for i in range(1024)))
    val_x = torch.stack(val_x, 0).to(device)
    val_l = torch.LongTensor(val_l).to(device)
    # Exclude the validation batch from the training data
    train_loader.dataset.data = train_loader.dataset.data[1024:]
    train_loader.dataset.targets = train_loader.dataset.targets[1024:]
    # Add the noise-augmentation to the (non-validation) training data:
    train_loader.dataset.transform = transforms.Compose([train_loader.dataset.transform, lambda x: x + args.cinn_sigma_noise * torch.randn_like(x)])

    t_start = time.time()
    nll_mean = []
    ndim_total = args.img_size * args.img_size


    print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
    val_nnl = []
    train_nnl = []
    for epoch in range(args.epochs):
        ### Training
        train_batch_nll = []
        for i, (x, l) in enumerate(train_loader):
            
            x, l = x.to(device), l.to(device)           
            z, log_j = cinn(x, l)

            nll = torch.mean(z**2) / 2 - torch.mean(log_j) / ndim_total
            # loss defined in subsection 3.2 of https://arxiv.org/pdf/1907.02392
            nll.backward()
            torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, 10.)
            nll_mean.append(nll.item())
            cinn.optimizer.step()
            cinn.optimizer.zero_grad()
            train_batch_nll.append(nll.item())

            if not i % args.print_freq:
                with torch.no_grad():
                    z, log_j = cinn(val_x, val_l)
                    nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / ndim_total

                print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
                                                                i, len(train_loader),
                                                                (time.time() - t_start)/60.,
                                                                np.mean(nll_mean),
                                                                nll_val.item(),
                                                                cinn.optimizer.param_groups[0]['lr'],
                                                                ), flush=True)
                nll_mean = []
            
        train_nnl.append(np.mean(train_batch_nll))
        scheduler.step()

        ### Testing        
        with torch.no_grad():
            cinn.eval()
            val_loss = []
            for i, (x, l) in enumerate(test_loader):
                x, l = x.to(device), l.to(device)

                z, log_j = cinn(x, l)

                nll = (torch.mean(z**2) / 2 - torch.mean(log_j)) / ndim_total
                val_loss.append(nll.item())
            print('Val loss: ', np.mean(val_loss), '--- Model path name: ', args.experiment_name)
            val_nnl.append(np.mean(val_loss))


        np.save('{}/{}_val_nll.npy'.format(exp_path, args.experiment_name), np.asarray(val_nnl))
        np.save('{}/{}_train_nll.npy'.format(exp_path, args.experiment_name), np.asarray(train_nnl))
        torch.save(cinn.state_dict(), "{}/{}.pt".format(exp_path, args.experiment_name))
