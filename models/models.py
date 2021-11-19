import torch.nn as nn
import torch
from models.SG import GeneratorUNet,Discriminator
from models.image_classification_net_64 import img_clif_net as img_clif_net_64
from models.image_classification_net_96 import img_clif_net as img_clif_net_96
from models.image_classification_net_128 import img_clif_net as img_clif_net_128
from models.vgg16 import VGG16
from models.Resnet18 import ResNet
####################################################
# Initialize generator and discriminator
####################################################
def Create_nets(args):
    generator = GeneratorUNet()
    discriminator = Discriminator(args)
    discriminator2=Discriminator(args)

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        discriminator2=discriminator2.cuda()
    if args.epoch_start != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load('log/%s-%s/%s/generator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,args.epoch_start)))
        discriminator.load_state_dict(torch.load('log/%s-%s/%s/discriminator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,args.epoch_start)))
        discriminator2.load_state_dict(torch.load('log/%s-%s/%s/discriminator2_%d.pth' % (
        args.exp_name, args.dataset_name, args.model_result_dir, args.epoch_start)))

    return generator, discriminator,discriminator2

def Create_nets_2(args):
    generator = GeneratorUNet()
    discriminator2= img_clif_net_64(3, 1)
    discriminator3 = img_clif_net_96(3, 1)
    discriminator4 = img_clif_net_128(3, 1)
    discriminator5 = VGG16()

    cuda_avail = torch.cuda.is_available()
    path2 = args.path_param_64
    path3 = args.path_param_96
    path4 = args.path_param_128
    path5 = args.path_vgg_256

    if cuda_avail:
        generator = generator.cuda()
        discriminator2.cuda()
        discriminator3.cuda()
        discriminator4.cuda()
        discriminator5.cuda()

    discriminator2.load_state_dict(torch.load(path2))
    discriminator3.load_state_dict(torch.load(path3))
    discriminator4.load_state_dict(torch.load(path4))
    discriminator5.load_state_dict(torch.load(path5)['model'])


    if torch.cuda.is_available():
        discriminator2 = discriminator2.cuda()
        discriminator3 = discriminator3.cuda()
        discriminator4 = discriminator4.cuda()
        discriminator5 = discriminator5.cuda()


    return generator,discriminator2,discriminator3,discriminator4,discriminator5
if __name__ == '__main__':
    c=Discriminator()
    print()