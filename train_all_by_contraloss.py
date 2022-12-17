import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
os.environ['CUDA_VISIBLE_DEVICES']='0'
from loss import *
from models.models import Create_nets,Create_nets_2
from datasets import *
from options import TrainOptions
from optimizer import *
from test import test
from eval import eval1
from utils import *
from models.vgg16 import VGG16
from models.image_classification_net_16 import img_clif_net as img_clif_net_16
from models.image_classification_net_32 import img_clif_net as img_clif_net_32
from models.image_classification_net_64 import img_clif_net as img_clif_net_64
from models.image_classification_net_96 import img_clif_net as img_clif_net_96
from models.image_classification_net_128 import img_clif_net as img_clif_net_128
device = "cuda" if torch.cuda.is_available() else "cpu"
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
import random
import numpy as np
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

#load the args
args = TrainOptions().parse()

# Initialize generator and discriminator
# generator, discriminator_1,discriminator2_1 = Create_nets(args)
generator, discriminator2, discriminator3, discriminator4, discriminator5 = Create_nets_2(args)

discriminator6 = img_clif_net_16(3, 1)
discriminator7 = img_clif_net_32(3, 1)
cuda_avail = torch.cuda.is_available()
if cuda_avail:
    discriminator6 = torch.load('./parameters/model_16_16.pth').cuda()
    discriminator7 = torch.load('./parameters/model_32_32.pth').cuda()
# discriminator6.load_state_dict(torch.load('./parameters/model_16_16.pth'))
# discriminator7.load_state_dict(torch.load('./parameters/model_32_32.pth'))


if args.epoch_start != 0:
    path = './log/SG-1204/saved_models/generator_all_by_contra_' + str(args.epoch_start) + '.pth'
    generator.load_state_dict(torch.load(path))
# Loss functions
criterion_GAN, criterion_pixelwise = Get_loss_func(args)
mask_loss_fn = MaskLoss(args.min_mask_coverage, args.mask_alpha, args.binarization_alpha)
# Optimizers
# optimizer_G, optimizer_D_1,optimizer_D2_1 = Get_optimizers(args, generator, discriminator_1,discriminator2_1)
optimizer_G = torch.optim.SGD(
    generator.parameters(),
    lr=args.lr, momentum=0.5)
optimizer_D2 = Adam(discriminator2.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer_D3 = Adam(discriminator3.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer_D4 = Adam(discriminator4.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer_D5 = Adam(discriminator5.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer_D6 = Adam(discriminator6.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer_D7 = Adam(discriminator7.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer_G = torch.optim.SGD(
#         generator.parameters(),
#         lr=args.lr*0.1, momentum=0.5)
log={'bestmae_it':0,'best_mae':10,'fm':0,'bestfm_it':0,'best_fm':0,'mae':0}
# Configure dataloaders
real_loder = Get_dataloader(args.path_gt, args.batch_size)
dis_C_loder1 = Get_dataloader(args.path_clear, args.batch_size)
dis_C_loder2 = Get_dataloader(args.path_clear, args.batch_size)
dis_C_loder3 = Get_dataloader(args.path_clear, args.batch_size)
dis_C_loder4 = Get_dataloader(args.path_clear, args.batch_size)

dis_B_loder1 = Get_dataloader(args.path_blur, args.batch_size)
dis_B_loder2 = Get_dataloader(args.path_blur, args.batch_size)
dis_B_loder3 = Get_dataloader(args.path_blur, args.batch_size)
dis_B_loder4 = Get_dataloader(args.path_blur, args.batch_size)

real = iter(real_loder)
dis_C1 = iter(dis_C_loder1)
dis_C2 = iter(dis_C_loder2)
dis_C3 = iter(dis_C_loder3)
dis_C4 = iter(dis_C_loder4)

dis_B1 = iter(dis_B_loder1)
dis_B2 = iter(dis_B_loder2)
dis_B3 = iter(dis_B_loder3)
dis_B4 = iter(dis_B_loder4)

j = 0

avg_loss = 0

cuda_avail = torch.cuda.is_available()
path2 = args.path_param_64
path3 = args.path_param_96
path4 = args.path_param_128
path5 = args.path_vgg_256

model_d2 = img_clif_net_64(3, 1)
model_d3 = img_clif_net_96(3, 1)
model_d4 = img_clif_net_128(3, 1)
model_d5 = VGG16()
model_d6 = torch.load('./parameters/model_16_16.pth').eval()
model_d7 = torch.load('./parameters/model_32_32.pth').eval()

for p in model_d2.parameters():
    p.requires_grad = False
for p in model_d3.parameters():
    p.requires_grad = False
for p in model_d4.parameters():
    p.requires_grad = False
for p in model_d5.parameters():
    p.requires_grad = False
for p in model_d6.parameters():
    p.requires_grad = False
for p in model_d7.parameters():
    p.requires_grad = False

model_d2.load_state_dict(torch.load(path2))
model_d2.eval()
model_d3.load_state_dict(torch.load(path3))
model_d3.eval()
model_d4.load_state_dict(torch.load(path4))
model_d4.eval()
model_d5.load_state_dict(torch.load(path5)['model'])
model_d5.eval()
# model_d6.load_state_dict(torch.load('./parameters/model_16_16.pth'))
# model_d6.eval()
# model_d7.load_state_dict(torch.load('./parameters/model_32_32.pth'))
# model_d7.eval()

if cuda_avail:
    model_d2.cuda()
    model_d3.cuda()
    model_d4.cuda()
    model_d5.cuda()
    model_d6.cuda()
    model_d7.cuda()

# 开始训练
pbar = range(args.epoch_start, 80000)
# pbar = range(args.epoch_start, 71680)
for i_num in pbar:

    try:
        real_image = next(real)
        real_image = real_image.cuda()

        dis_C_image1 = next(dis_C1)
        dis_C_image1 = dis_C_image1.cuda()
        dis_C_image2 = next(dis_C2)
        dis_C_image2 = dis_C_image2.cuda()
        dis_C_image3 = next(dis_C3)
        dis_C_image3 = dis_C_image3.cuda()
        dis_C_image4 = next(dis_C4)
        dis_C_image4 = dis_C_image4.cuda()

        dis_B_image1 = next(dis_B1)
        dis_B_image1 = dis_B_image1.cuda()
        dis_B_image2 = next(dis_B2)
        dis_B_image2 = dis_B_image2.cuda()
        dis_B_image3 = next(dis_B3)
        dis_B_image3 = dis_B_image3.cuda()
        dis_B_image4 = next(dis_B4)
        dis_B_image4 = dis_B_image4.cuda()


    except (OSError, StopIteration):

        real = iter(real_loder)
        dis_C1 = iter(dis_C_loder1)
        dis_C2 = iter(dis_C_loder2)
        dis_C3 = iter(dis_C_loder3)
        dis_C4 = iter(dis_C_loder4)

        dis_B1 = iter(dis_B_loder1)
        dis_B2 = iter(dis_B_loder2)
        dis_B3 = iter(dis_B_loder3)
        dis_B4 = iter(dis_B_loder4)

        real_image = next(real)
        real_image = real_image.to(device)

        dis_C_image1 = next(dis_C1)
        dis_C_image1 = dis_C_image1.to(device)
        dis_C_image2 = next(dis_C2)
        dis_C_image2 = dis_C_image2.to(device)
        dis_C_image3 = next(dis_C3)
        dis_C_image3 = dis_C_image3.to(device)
        dis_C_image4 = next(dis_C4)
        dis_C_image4 = dis_C_image4.to(device)

        dis_B_image1 = next(dis_B1)
        dis_B_image1 = dis_B_image1.to(device)
        dis_B_image2 = next(dis_B2)
        dis_B_image2 = dis_B_image2.to(device)
        dis_B_image3 = next(dis_B3)
        dis_B_image3 = dis_B_image3.to(device)
        dis_B_image4 = next(dis_B4)
        dis_B_image4 = dis_B_image4.to(device)

    # ------------------
    #  Train Generators
    # ------------------

    # Adversarial ground truths
    patch=(1,1,1)
    valid = Variable(torch.FloatTensor(np.ones((real_image.size(0),*patch))).cuda(), requires_grad=False)
    fake = Variable(torch.FloatTensor(np.zeros((real_image.size(0),*patch))).cuda(), requires_grad=False)

    optimizer_G.zero_grad()
    requires_grad(generator, True)
    requires_grad(discriminator2, False)
    requires_grad(discriminator3, False)
    requires_grad(discriminator4, False)
    requires_grad(discriminator5, False)

    mask = generator(real_image)
    Mask1=mask
    # syn image
    syn_image_clear = Mask1 * real_image + (1 - Mask1) * dis_C_image1
    syn_image_blur = Mask1 * dis_B_image1 + (1 - Mask1) * real_image


    loss_mask, _ = mask_loss_fn(mask)
    #     loss_16_d = feather_loss_d6(syn_image_clear, dis_C_image2, syn_image_blur, dis_B_image2, discriminator6)
    loss_32_d = feather_loss_d7(syn_image_clear, dis_C_image2, syn_image_blur, dis_B_image2, discriminator7)
    loss_64_d = feather_loss_d2(syn_image_clear, dis_C_image2, syn_image_blur, dis_B_image2, discriminator2)
    loss_96_d = feather_loss_d3(syn_image_clear, dis_C_image2, syn_image_blur, dis_B_image2, discriminator3)
    loss_128_d = feather_loss_d4(syn_image_clear, dis_C_image2, syn_image_blur, dis_B_image2, discriminator4)
    loss_global_d = feather_loss_d5(syn_image_clear, dis_C_image2, syn_image_blur, dis_B_image2, discriminator5)

    zeta_128 = 1.0
    zeta_96 = 1.0
    zeta_64 = 1.0
    # Total loss
    # loss_D = loss_c_64_d + loss_b_64_d + loss_c_96_d + loss_b_96_d + loss_c_128_d + loss_b_128_d + loss_c_global_d + loss_b_global_d

    loss_D = zeta_64 * loss_64_d + zeta_96 * loss_96_d + zeta_128 * loss_128_d + loss_global_d + loss_32_d
    loss_G = loss_mask + loss_D
    # loss_G=loss_GAN2+loss_mask+loss_GAN1
    loss_G.backward()
    optimizer_G.step()

    # if i_num < 400:
    if i_num < 64000:
        if i_num % 400 == 0:
            image_path = 'log/%s-%s/%s' % (args.exp_name, args.dataset_name, args.img_result_dir)
            os.makedirs(image_path, exist_ok=True)
            to_image(real_image, i=i_num, tag='input', path=image_path)
            to_image(syn_image_clear, i=i_num, tag='syn_image', path=image_path)
            to_image(syn_image_blur, i=i_num, tag='syn_blur', path=image_path)
            to_image_mask(mask, i=i_num, tag='mask', path=image_path)

        if args.checkpoint_interval != -1 and i_num % 200 == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), 'log/%s-%s/%s/generator_all_by_contra_%d.pth' % (
                args.exp_name, args.dataset_name, args.model_result_dir, i_num))
            # torch.save(discriminator.state_dict(), 'log/%s-%s/%s/discriminator1_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir, i))
            # torch.save(discriminator2.state_dict(), 'log/%s-%s/%s/discriminator2_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir, i))

            pthpath = 'log/%s-%s/%s/generator_all_by_contra_%d.pth' % (
                args.exp_name, args.dataset_name, args.model_result_dir, i_num)
            # print(pthpath)
            mask_save_path = 'log/%s-%s/test/test100-%s' % (args.exp_name, args.dataset_name, i_num)
            # print(mask_save_path)
            os.makedirs(mask_save_path, exist_ok=True)

            image_path = './dataset/test/xu100-source'
            test(pthpath, mask_save_path, image_path)

            gt_path = './dataset/test/xu100-gt'
            mae1, fmeasure1, _, _ = eval1(mask_save_path, gt_path, 1.5)

            if mae1 < log['best_mae']:
                log['bestmae_it'] = i_num
                log['best_mae'] = mae1
                log['fm'] = fmeasure1
            if fmeasure1 > log['best_fm']:
                log['bestfm_it'] = i_num
                log['best_fm'] = fmeasure1
                log['mae'] = mae1
            print(
                '====================================================================================================================')
            print('batch:', i_num, "mae:", mae1, "fmeasure:", fmeasure1)
            print('bestmae_it', log['bestmae_it'], 'best_mae', log['best_mae'], 'fm:', log['fm'])
            print('bestfm_it', log['bestfm_it'], 'mae:', log['mae'], 'best_fm', log['best_fm'])
            print(
                '=====================================================================================================================')
        continue

    # ---------------------
    #  Train Discriminator
    # # ---------------------
    optimizer_D2.zero_grad()
    requires_grad(generator, False)
    requires_grad(discriminator2, True)
    requires_grad(discriminator3, False)
    requires_grad(discriminator4, False)
    requires_grad(discriminator5, False)
    requires_grad(discriminator6, False)
    requires_grad(discriminator7, False)

    syn_C_image_1 = dis_C_image3 * Mask1 + dis_C_image4 * (1 - Mask1)
    syn_B_image_1 = dis_B_image3 * Mask1 + dis_B_image4 * (1 - Mask1)

    syn_image_clear_detach = syn_image_clear.detach()
    syn_image_blur_detach = syn_image_blur.detach()
    syn_image_clear_for_compare = syn_C_image_1.detach()
    syn_image_blur_for_compare = syn_B_image_1.detach()

    syn_clear_image_list = []
    clear_image_list = []
    syn_blur_image_list = []
    blur_image_list = []
    num_block = 13 * 13
    for i in range(13):
        for j in range(13):
            w = i * 16
            k = j * 16
            syn_clear_image_list.append(syn_image_clear_detach[:, :, w:w + 64, k:k + 64])
            clear_image_list.append(syn_image_clear_for_compare[:, :, w:w + 64, k:k + 64])
            syn_blur_image_list.append(syn_image_blur_detach[:, :, w:w + 64, k:k + 64])
            blur_image_list.append(syn_image_blur_for_compare[:, :, w:w + 64, k:k + 64])

    syn_clear_image = torch.cat(syn_clear_image_list, dim=0)
    clear_image = torch.cat(clear_image_list, dim=0)
    syn_blur_image = torch.cat(syn_blur_image_list, dim=0)
    blur_image = torch.cat(blur_image_list, dim=0)

    feather_syn_c, output_c = discriminator2(syn_clear_image)
    feather_c, output_cc = discriminator2(clear_image)

    feather_syn_b, output_b = discriminator2(syn_blur_image)
    feather_b, output_bb = discriminator2(blur_image)

    feather_syn_c = rearrange(feather_syn_c, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_c = rearrange(feather_c, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_syn_b = rearrange(feather_syn_b, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_b = rearrange(feather_b, '(b1 b2) c -> b1 b2 c', b1=num_block)

    c_fea1 = torch.cat((feather_c, feather_syn_c), dim=1)
    c_fea1 = c_fea1.unsqueeze(1)
    c_fea = torch.cat((c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1), dim=1)
    # c_fea_no_diag = c_fea[~torch.eye(c_fea.shape[0],dtype=bool)].reshape((8,7,120),-1)
    c_fea_replace = c_fea.permute(0, 2, 1, 3)
    b_fea1 = torch.cat((feather_b, feather_syn_b), dim=1)
    b_fea1 = b_fea1.unsqueeze(1)
    b_fea = torch.cat((b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1), dim=1)
    b_fea_replace = b_fea.permute(0, 2, 1, 3)
    c_fea_no_diag = c_fea[:, ~torch.eye(c_fea.shape[1], dtype=bool)]
    c_fea_replace_no_diag = c_fea_replace[:, ~torch.eye(c_fea_replace.shape[1], dtype=bool)]
    similarity1 = torch.exp(torch.cosine_similarity(c_fea_no_diag, c_fea_replace_no_diag, dim=2))
    b_fea_no_diag = b_fea[:, ~torch.eye(b_fea.shape[1], dtype=bool)]
    b_fea_replace_no_diag = b_fea_replace[:, ~torch.eye(b_fea_replace.shape[1], dtype=bool)]
    similarity2 = torch.exp(torch.cosine_similarity(b_fea_no_diag, b_fea_replace_no_diag, dim=2))
    unsimilarity = torch.exp(torch.cosine_similarity(c_fea, b_fea, dim=3))

    sum_s1 = torch.sum(similarity1, dim=1)
    sum_s2 = torch.sum(similarity2, dim=1)
    sum_us = torch.sum(unsimilarity, dim=(1, 2))
    loss = -torch.mean(torch.log(sum_s1 / (sum_s1 + sum_us)) + torch.log(sum_s2 / (sum_s2 + sum_us)))
    loss.backward()
    # 根据计算的梯度调整参数
    optimizer_D2.step()


    optimizer_D3.zero_grad()
    requires_grad(generator, False)

    requires_grad(discriminator2, False)
    requires_grad(discriminator3, True)
    requires_grad(discriminator4, False)
    requires_grad(discriminator5, False)
    requires_grad(discriminator6, False)
    requires_grad(discriminator7, False)

    syn_clear_image_list = []
    clear_image_list = []
    syn_blur_image_list = []
    blur_image_list = []
    num_block = 11 * 11
    for i in range(11):
        for j in range(11):
            w = i * 16
            k = j * 16
            syn_clear_image_list.append(syn_image_clear_detach[:, :, w:w + 96, k:k + 96])
            clear_image_list.append(syn_image_clear_for_compare[:, :, w:w + 96, k:k + 96])
            syn_blur_image_list.append(syn_image_blur_detach[:, :, w:w + 96, k:k + 96])
            blur_image_list.append(syn_image_blur_for_compare[:, :, w:w + 96, k:k + 96])

    syn_clear_image = torch.cat(syn_clear_image_list, dim=0)
    clear_image = torch.cat(clear_image_list, dim=0)
    syn_blur_image = torch.cat(syn_blur_image_list, dim=0)
    blur_image = torch.cat(blur_image_list, dim=0)

    feather_syn_c, output_c = discriminator3(syn_clear_image)
    feather_c, output_cc = discriminator3(clear_image)

    feather_syn_b, output_b = discriminator3(syn_blur_image)
    feather_b, output_bb = discriminator3(blur_image)

    feather_syn_c = rearrange(feather_syn_c, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_c = rearrange(feather_c, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_syn_b = rearrange(feather_syn_b, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_b = rearrange(feather_b, '(b1 b2) c -> b1 b2 c', b1=num_block)

    c_fea1 = torch.cat((feather_c, feather_syn_c), dim=1)
    c_fea1 = c_fea1.unsqueeze(1)
    c_fea = torch.cat((c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1), dim=1)
    # c_fea_no_diag = c_fea[~torch.eye(c_fea.shape[0],dtype=bool)].reshape((8,7,120),-1)
    c_fea_replace = c_fea.permute(0, 2, 1, 3)
    b_fea1 = torch.cat((feather_b, feather_syn_b), dim=1)
    b_fea1 = b_fea1.unsqueeze(1)
    b_fea = torch.cat((b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1), dim=1)
    b_fea_replace = b_fea.permute(0, 2, 1, 3)
    c_fea_no_diag = c_fea[:, ~torch.eye(c_fea.shape[1], dtype=bool)]
    c_fea_replace_no_diag = c_fea_replace[:, ~torch.eye(c_fea_replace.shape[1], dtype=bool)]
    similarity1 = torch.exp(torch.cosine_similarity(c_fea_no_diag, c_fea_replace_no_diag, dim=2))
    b_fea_no_diag = b_fea[:, ~torch.eye(b_fea.shape[1], dtype=bool)]
    b_fea_replace_no_diag = b_fea_replace[:, ~torch.eye(b_fea_replace.shape[1], dtype=bool)]
    similarity2 = torch.exp(torch.cosine_similarity(b_fea_no_diag, b_fea_replace_no_diag, dim=2))
    unsimilarity = torch.exp(torch.cosine_similarity(c_fea, b_fea, dim=3))

    sum_s1 = torch.sum(similarity1, dim=1)
    sum_s2 = torch.sum(similarity2, dim=1)
    sum_us = torch.sum(unsimilarity, dim=(1, 2))
    loss = -torch.mean(torch.log(sum_s1 / (sum_s1 + sum_us)) + torch.log(sum_s2 / (sum_s2 + sum_us)))
    loss.backward()
    # 根据计算的梯度调整参数

    optimizer_D3.step()


    optimizer_D4.zero_grad()
    requires_grad(generator, False)
    requires_grad(discriminator2, False)
    requires_grad(discriminator3, False)
    requires_grad(discriminator4, True)
    requires_grad(discriminator5, False)
    requires_grad(discriminator6, False)
    requires_grad(discriminator7, False)

    syn_clear_image_list = []
    clear_image_list = []
    syn_blur_image_list = []
    blur_image_list = []
    num_block = 9 * 9
    for i in range(9):
        for j in range(9):
            w = i * 16
            k = j * 16
            syn_clear_image_list.append(syn_image_clear_detach[:, :, w:w + 128, k:k + 128])
            clear_image_list.append(syn_image_clear_for_compare[:, :, w:w + 128, k:k + 128])
            syn_blur_image_list.append(syn_image_blur_detach[:, :, w:w + 128, k:k + 128])
            blur_image_list.append(syn_image_blur_for_compare[:, :, w:w + 128, k:k + 128])

    syn_clear_image = torch.cat(syn_clear_image_list, dim=0)
    clear_image = torch.cat(clear_image_list, dim=0)
    syn_blur_image = torch.cat(syn_blur_image_list, dim=0)
    blur_image = torch.cat(blur_image_list, dim=0)

    feather_syn_c, output_c = discriminator4(syn_clear_image)
    feather_c, output_cc = discriminator4(clear_image)

    feather_syn_b, output_b = discriminator4(syn_blur_image)
    feather_b, output_bb = discriminator4(blur_image)

    feather_syn_c = rearrange(feather_syn_c, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_c = rearrange(feather_c, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_syn_b = rearrange(feather_syn_b, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_b = rearrange(feather_b, '(b1 b2) c -> b1 b2 c', b1=num_block)

    c_fea1 = torch.cat((feather_c, feather_syn_c), dim=1)
    c_fea1 = c_fea1.unsqueeze(1)
    c_fea = torch.cat((c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1), dim=1)
    # c_fea_no_diag = c_fea[~torch.eye(c_fea.shape[0],dtype=bool)].reshape((8,7,120),-1)
    c_fea_replace = c_fea.permute(0, 2, 1, 3)
    b_fea1 = torch.cat((feather_b, feather_syn_b), dim=1)
    b_fea1 = b_fea1.unsqueeze(1)
    b_fea = torch.cat((b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1), dim=1)
    b_fea_replace = b_fea.permute(0, 2, 1, 3)
    c_fea_no_diag = c_fea[:, ~torch.eye(c_fea.shape[1], dtype=bool)]
    c_fea_replace_no_diag = c_fea_replace[:, ~torch.eye(c_fea_replace.shape[1], dtype=bool)]
    similarity1 = torch.exp(torch.cosine_similarity(c_fea_no_diag, c_fea_replace_no_diag, dim=2))
    b_fea_no_diag = b_fea[:, ~torch.eye(b_fea.shape[1], dtype=bool)]
    b_fea_replace_no_diag = b_fea_replace[:, ~torch.eye(b_fea_replace.shape[1], dtype=bool)]
    similarity2 = torch.exp(torch.cosine_similarity(b_fea_no_diag, b_fea_replace_no_diag, dim=2))
    unsimilarity = torch.exp(torch.cosine_similarity(c_fea, b_fea, dim=3))

    sum_s1 = torch.sum(similarity1, dim=1)
    sum_s2 = torch.sum(similarity2, dim=1)
    sum_us = torch.sum(unsimilarity, dim=(1, 2))
    loss = -torch.mean(torch.log(sum_s1 / (sum_s1 + sum_us)) + torch.log(sum_s2 / (sum_s2 + sum_us)))
    loss.backward()
    # 根据计算的梯度调整参数
    optimizer_D4.step()

    optimizer_D5.zero_grad()
    requires_grad(generator, False)
    requires_grad(discriminator2, False)
    requires_grad(discriminator3, False)
    requires_grad(discriminator4, False)
    requires_grad(discriminator5, True)
    requires_grad(discriminator6, False)
    requires_grad(discriminator7, False)

    cfeature1, _ = discriminator5(syn_image_clear_for_compare)
    bfeature1, _ = discriminator5(syn_image_blur_for_compare)
    syn_cfeature, _ = discriminator5(syn_image_clear_detach)
    syn_bfeature, _ = discriminator5(syn_image_blur_detach)

    c_fea1 = torch.cat((cfeature1, syn_cfeature), dim=0)
    c_fea1 = c_fea1.unsqueeze(0)
    c_fea = torch.cat((c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1), dim=0)
    # c_fea_no_diag = c_fea[~torch.eye(c_fea.shape[0],dtype=bool)].reshape((8,7,120),-1)
    c_fea_replace = c_fea.permute(1, 0, 2)
    b_fea1 = torch.cat((bfeature1, syn_bfeature), dim=0)
    b_fea1 = b_fea1.unsqueeze(0)
    b_fea = torch.cat((b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1), dim=0)
    b_fea_replace = b_fea.permute(1, 0, 2)
    c_fea_no_diag = c_fea[~torch.eye(c_fea.shape[0], dtype=bool)]
    c_fea_replace_no_diag = c_fea_replace[~torch.eye(c_fea_replace.shape[0], dtype=bool)]
    similarity1 = torch.exp(torch.cosine_similarity(c_fea_no_diag, c_fea_replace_no_diag, dim=1))
    b_fea_no_diag = b_fea[~torch.eye(b_fea.shape[0], dtype=bool)]
    b_fea_replace_no_diag = b_fea_replace[~torch.eye(b_fea_replace.shape[0], dtype=bool)]
    similarity2 = torch.exp(torch.cosine_similarity(b_fea_no_diag, b_fea_replace_no_diag, dim=1))
    unsimilarity = torch.exp(torch.cosine_similarity(c_fea, b_fea, dim=2))

    sum_s1 = torch.sum(similarity1, dim=0)
    sum_s2 = torch.sum(similarity2, dim=0)
    sum_us = torch.sum(unsimilarity, dim=(0, 1))
    loss = -torch.mean(torch.log(sum_s1 / (sum_s1 + sum_us)) + torch.log(sum_s2 / (sum_s2 + sum_us)))
    loss.backward()
    # 根据计算的梯度调整参数
    optimizer_D5.step()


    optimizer_D7.zero_grad()
    requires_grad(generator, False)
    requires_grad(discriminator2, False)
    requires_grad(discriminator3, False)
    requires_grad(discriminator4, False)
    requires_grad(discriminator5, False)
    requires_grad(discriminator6, False)
    requires_grad(discriminator7, True)

    syn_clear_image_list = []
    clear_image_list = []
    syn_blur_image_list = []
    blur_image_list = []
    num_block = 15 * 15
    for i in range(15):
        for j in range(15):
            w = i * 16
            k = j * 16
            syn_clear_image_list.append(syn_image_clear_detach[:, :, w:w + 32, k:k + 32])
            clear_image_list.append(syn_image_clear_for_compare[:, :, w:w + 32, k:k + 32])
            syn_blur_image_list.append(syn_image_blur_detach[:, :, w:w + 32, k:k + 32])
            blur_image_list.append(syn_image_blur_for_compare[:, :, w:w + 32, k:k + 32])

    syn_clear_image = torch.cat(syn_clear_image_list, dim=0)
    clear_image = torch.cat(clear_image_list, dim=0)
    syn_blur_image = torch.cat(syn_blur_image_list, dim=0)
    blur_image = torch.cat(blur_image_list, dim=0)

    feather_syn_c, output_c = discriminator7(syn_clear_image)
    feather_c, output_cc = discriminator7(clear_image)

    feather_syn_b, output_b = discriminator7(syn_blur_image)
    feather_b, output_bb = discriminator7(blur_image)

    feather_syn_c = rearrange(feather_syn_c, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_c = rearrange(feather_c, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_syn_b = rearrange(feather_syn_b, '(b1 b2) c -> b1 b2 c', b1=num_block)
    feather_b = rearrange(feather_b, '(b1 b2) c -> b1 b2 c', b1=num_block)

    c_fea1 = torch.cat((feather_c, feather_syn_c), dim=1)
    c_fea1 = c_fea1.unsqueeze(1)
    c_fea = torch.cat((c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1), dim=1)
    # c_fea_no_diag = c_fea[~torch.eye(c_fea.shape[0],dtype=bool)].reshape((8,7,120),-1)
    c_fea_replace = c_fea.permute(0, 2, 1, 3)
    b_fea1 = torch.cat((feather_b, feather_syn_b), dim=1)
    b_fea1 = b_fea1.unsqueeze(1)
    b_fea = torch.cat((b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1), dim=1)
    b_fea_replace = b_fea.permute(0, 2, 1, 3)
    c_fea_no_diag = c_fea[:, ~torch.eye(c_fea.shape[1], dtype=bool)]
    c_fea_replace_no_diag = c_fea_replace[:, ~torch.eye(c_fea_replace.shape[1], dtype=bool)]
    similarity1 = torch.exp(torch.cosine_similarity(c_fea_no_diag, c_fea_replace_no_diag, dim=2))
    b_fea_no_diag = b_fea[:, ~torch.eye(b_fea.shape[1], dtype=bool)]
    b_fea_replace_no_diag = b_fea_replace[:, ~torch.eye(b_fea_replace.shape[1], dtype=bool)]
    similarity2 = torch.exp(torch.cosine_similarity(b_fea_no_diag, b_fea_replace_no_diag, dim=2))
    unsimilarity = torch.exp(torch.cosine_similarity(c_fea, b_fea, dim=3))

    sum_s1 = torch.sum(similarity1, dim=1)
    sum_s2 = torch.sum(similarity2, dim=1)
    sum_us = torch.sum(unsimilarity, dim=(1, 2))
    loss = -torch.mean(torch.log(sum_s1 / (sum_s1 + sum_us)) + torch.log(sum_s2 / (sum_s2 + sum_us)))
    loss.backward()
    # 根据计算的梯度调整参数
    optimizer_D7.step()

    if i_num % 50==0:
        image_path = 'log/%s-%s/%s' % (args.exp_name, args.dataset_name, args.img_result_dir)
        os.makedirs(image_path, exist_ok=True)
        to_image(real_image, i=i_num, tag='input', path=image_path)
        to_image(syn_image_clear, i=i_num, tag='syn_image', path=image_path)
        to_image(syn_image_blur, i=i_num, tag='syn_blur', path=image_path)
        to_image_mask(mask, i=i_num, tag='mask', path=image_path)

    if args.checkpoint_interval != -1 and i_num % 20== 0:
    # Save model checkpoints
        torch.save(generator.state_dict(), 'log/%s-%s/%s/generator_all_by_contra_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,i_num))
        # torch.save(discriminator.state_dict(), 'log/%s-%s/%s/discriminator1_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir, i))
        # torch.save(discriminator2.state_dict(), 'log/%s-%s/%s/discriminator2_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir, i))

        pthpath='log/%s-%s/%s/generator_all_by_contra_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,i_num)
        mask_save_path = 'log/%s-%s/test/test100-%s' % (args.exp_name, args.dataset_name, i_num)
        image_path= './dataset/test/xu100-source'
        test(pthpath,mask_save_path,image_path)

        gt_path = './dataset/test/xu100-gt'
        mae1,fmeasure1,_,_=eval1(mask_save_path,gt_path,1.5)

        if mae1<log['best_mae'] :
            log['bestmae_it']=i_num
            log['best_mae']=mae1
            log['fm']=fmeasure1
        if fmeasure1>log['best_fm']:
            log['bestfm_it']=i_num
            log['best_fm']=fmeasure1
            log['mae']=mae1
        print('====================================================================================================================')
        print('batch:',i_num, "mae:", mae1, "fmeasure:", fmeasure1)
        print('bestmae_it',log['bestmae_it'],'best_mae',log['best_mae'],'fm:',log['fm'])
        print('bestfm_it',log['bestfm_it'],'mae:',log['mae'],'best_fm',log['best_fm'])
        print('=====================================================================================================================')