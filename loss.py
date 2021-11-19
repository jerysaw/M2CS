#mask损失
import torch
import torch
import torch.nn.functional as F
from functools import partial
from models.vgg16 import VGG16
from models.Resnet18 import ResNet
from models.image_classification_net_64 import img_clif_net as img_clif_net_64
from models.image_classification_net_96 import img_clif_net as img_clif_net_96
from models.image_classification_net_128 import img_clif_net as img_clif_net_128
from options import TrainOptions
from einops import rearrange
args = TrainOptions().parse()
def min_permask_loss(mask, min_mask_coverage):
    '''
    One object mask per channel in this case
    '''
    return F.relu(min_mask_coverage - mask.mean(dim=(2, 3))).mean()

def binarization_loss(mask):
    return torch.min(1-mask, mask).mean()


class MaskLoss:
    def __init__(self, min_mask_coverage, mask_alpha, bin_alpha, min_mask_fn=min_permask_loss):
        self.min_mask_coverage = min_mask_coverage
        self.mask_alpha = mask_alpha
        self.bin_alpha = bin_alpha
        self.min_mask_fn = partial(min_mask_fn, min_mask_coverage=min_mask_coverage)

    def __call__(self, mask):
        if type(mask) in (tuple, list):
            mask = torch.cat(mask, dim=1)
        min_loss = self.min_mask_fn(mask)
        bin_loss = binarization_loss(mask)
        # return self.mask_alpha * min_loss + self.bin_alpha * bin_loss, dict(min_mask_loss=min_loss, bin_loss=bin_loss)
        # return self.mask_alpha * min_loss, dict(min_mask_loss=min_loss, bin_loss=bin_loss)
        return self.mask_alpha * min_loss + self.bin_alpha * bin_loss, dict(min_mask_loss=min_loss, bin_loss=bin_loss)
#重构损失
def Loss1(rendered,real):
    loss_fn = torch.nn.L1Loss(reduce=True,size_average=True)
    loss = loss_fn(rendered,real)
    return loss

def Get_loss_func(args):
    device = torch.device("cuda:0")
    criterion_GAN = torch.nn.BCELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    if torch.cuda.is_available():

        # criterion_GAN.cuda()
        # criterion_pixelwise.cuda()
        criterion_GAN.to(device)
        criterion_pixelwise.to(device)
    return criterion_GAN, criterion_pixelwise

def feather_loss(syn_celar,clear,syn_blur,blur):
    cuda_avail = torch.cuda.is_available()
    model = VGG16()
    for p in model.parameters():
        p.requires_grad = False

    if cuda_avail:
        model.cuda()
    # load model
    path = args.path_vgg_256
    model_dict = torch.load(path)
    model.load_state_dict(model_dict['model'])
    model.eval()

    feather_syn_c, output_c = model(syn_celar)
    feather_c,output_cc = model(clear)
    # print(feather_syn_c.size())
    similarity=torch.cosine_similarity(feather_c,feather_syn_c,dim=1)
    loss_c=torch.mean(1-similarity,dim=0)
    # loss_c=l1loss(feather_syn_c,feather_c)

    feather_syn_b, output_b = model(syn_blur)
    feather_b, output_bb = model(blur)

    similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=1)
    loss_b = torch.mean(1 - similarity2, dim=0)
    # loss_b=l1loss(feather_syn_b,feather_b)

    _, predictionb = torch.max(output_bb.data, 1)#0
    # _, predictionbb = torch.max(output_bb.data, 1)  #
    _, predictionc = torch.max(output_cc.data, 1)#2
    nb=0
    for i in range(0,len(predictionb)):
        if predictionb[i]==0:
            nb+=1
    nc = 0
    for i in range(0, len(predictionc)):
        if predictionc[i] == 2:
            nc += 1
    return loss_c,loss_b,nc,nb

def feather_loss_resnet(syn_celar,clear,syn_blur,blur, model):

    feather_syn_c, output_c = model(syn_celar)
    feather_c,output_cc = model(clear)
    # print(feather_syn_c.size())
    similarity=torch.cosine_similarity(feather_c,feather_syn_c,dim=1)
    loss_c=torch.mean(1-similarity,dim=0)
    # loss_c=l1loss(feather_syn_c,feather_c)

    feather_syn_b, output_b = model(syn_blur)
    feather_b, output_bb = model(blur)

    similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=1)
    loss_b = torch.mean(1 - similarity2, dim=0)
    # loss_b=l1loss(feather_syn_b,feather_b)

    # _, predictionb = torch.max(output_b.data, 1)#0
    # _, predictionbb = torch.max(output_bb.data, 1)  #
    # _, predictionc = torch.max(output_c.data, 1)#2
    # nb=0
    # for i in range(0,len(predictionb)):
    #     if predictionb[i]==0:
    #         nb+=1
    # nc = 0
    # for i in range(0, len(predictionc)):
    #     if predictionc[i] == 2:
    #         nc += 1
    return loss_c,loss_b


def feather_loss_contriLoss(syn_celar,clear,syn_blur,blur):
    cuda_avail = torch.cuda.is_available()
    model = VGG16()
    for p in model.parameters():
        p.requires_grad = False

    if cuda_avail:
        model.cuda()
    # load model
    path = 'parameters/VGG16model_99'
    model_dict = torch.load(path)
    model.load_state_dict(model_dict['model'])
    model.eval()

    syn_cfeature, output_c = model(syn_celar)
    cfeature1,output_cc = model(clear)

    # similarity=torch.cosine_similarity(feather_c,feather_syn_c,dim=1)
    # loss_c=torch.mean(1-similarity,dim=0)

    syn_bfeature, output_b = model(syn_blur)
    bfeature1, output_bb = model(blur)

    # similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=1)
    # loss_b = torch.mean(1 - similarity2, dim=0)

    c_fea1 = torch.cat((cfeature1, syn_cfeature), dim=0)
    c_fea1 = c_fea1.unsqueeze(0)
    c_fea = torch.cat((c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1), dim=0)
    # c_fea_no_diag = c_fea[~torch.eye(c_fea.shape[0],dtype=bool)].reshape((8,7,120),-1)
    c_fea_replace = c_fea.permute(1, 0, 2)
    b_fea1 = torch.cat((bfeature1, syn_bfeature), dim=0)
    b_fea1 = b_fea1.unsqueeze(0)
    b_fea = torch.cat((b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1, b_fea1), dim=0)
    b_fea_replace = b_fea.permute(1, 0, 2)
    c_fea_no_diag = c_fea[~torch.eye(c_fea.shape[0], dtype=bool)].reshape((8, 7, -1), -1)
    c_fea_replace_no_diag = c_fea_replace[~torch.eye(c_fea_replace.shape[0], dtype=bool)].reshape((8, 7, -1), -1)
    similarity1 = torch.exp(torch.cosine_similarity(c_fea_no_diag, c_fea_replace_no_diag, dim=2))
    b_fea_no_diag = b_fea[~torch.eye(b_fea.shape[0], dtype=bool)].reshape((8, 7, -1), -1)
    b_fea_replace_no_diag = b_fea_replace[~torch.eye(b_fea_replace.shape[0], dtype=bool)].reshape((8, 7, -1), -1)
    similarity2 = torch.exp(torch.cosine_similarity(b_fea_no_diag, b_fea_replace_no_diag, dim=2))
    unsimilarity = torch.exp(torch.cosine_similarity(c_fea, b_fea, dim=2))
    loss = (-torch.log((torch.sum(similarity1) + torch.sum(similarity2)) / (
                torch.sum(similarity1) + torch.sum(similarity2) + torch.sum(unsimilarity)))).mean()
    return loss

#128的固定权重
def feather_loss_128(syn_celar,clear,syn_blur,blur, model):

    syn_clear_image_list = []
    clear_image_list = []
    syn_blur_image_list = []
    blur_image_list = []
    for i in range(9):
        for j in range(9):
            w = i * 16
            k = j * 16
            syn_clear_image_list.append(syn_celar[:, :, w:w + 128, k:k + 128])
            clear_image_list.append(clear[:, :, w:w + 128, k:k + 128])
            syn_blur_image_list.append(syn_blur[:, :, w:w + 128, k:k + 128])
            blur_image_list.append(blur[:, :, w:w + 128, k:k + 128])

    syn_clear_image = torch.cat(syn_clear_image_list, dim=0)
    clear_image = torch.cat(clear_image_list, dim=0)
    syn_blur_image = torch.cat(syn_blur_image_list, dim=0)
    blur_image = torch.cat(blur_image_list, dim=0)

    feather_syn_c, output_c = model(syn_clear_image)
    feather_c, output_cc = model(clear_image)

    feather_syn_b, output_b = model(syn_blur_image)
    feather_b, output_bb = model(blur_image)

    feather_syn_c = rearrange(feather_syn_c, '(b1 b2) c -> b1 b2 c', b1=9 * 9)
    feather_c = rearrange(feather_c, '(b1 b2) c -> b1 b2 c', b1=9 * 9)
    feather_syn_b = rearrange(feather_syn_b, '(b1 b2) c -> b1 b2 c', b1=9 * 9)
    feather_b = rearrange(feather_b, '(b1 b2) c -> b1 b2 c', b1=9 * 9)

    similarity1 = torch.cosine_similarity(feather_c, feather_syn_c, dim=2)
    similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=2)

    loss_c = torch.mean(1 - similarity1)
    loss_b = torch.mean(1 - similarity2)

    return loss_c,loss_b

#96的固定权重
def feather_loss_96(syn_celar,clear,syn_blur,blur, model):

    syn_clear_image_list = []
    clear_image_list = []
    syn_blur_image_list = []
    blur_image_list = []
    for i in range(11):
        for j in range(11):
            w = i * 16
            k = j * 16
            syn_clear_image_list.append(syn_celar[:,:,w:w + 96, k:k + 96])
            clear_image_list.append(clear[:,:,w:w + 96, k:k + 96])

            syn_blur_image_list.append(syn_blur[:, :, w:w + 96, k:k + 96])
            blur_image_list.append(blur[:, :, w:w + 96, k:k + 96])

    syn_clear_image = torch.cat(syn_clear_image_list, dim=0)
    clear_image = torch.cat(clear_image_list, dim=0)
    syn_blur_image = torch.cat(syn_blur_image_list, dim=0)
    blur_image = torch.cat(blur_image_list, dim=0)

    feather_syn_c, output_c = model(syn_clear_image)
    feather_c, output_cc = model(clear_image)

    feather_syn_b, output_b = model(syn_blur_image)
    feather_b, output_bb = model(blur_image)

    feather_syn_c = rearrange(feather_syn_c, '(b1 b2) c -> b1 b2 c', b1=11 * 11)
    feather_c = rearrange(feather_c, '(b1 b2) c -> b1 b2 c', b1=11 * 11)
    feather_syn_b = rearrange(feather_syn_b, '(b1 b2) c -> b1 b2 c', b1=11 * 11)
    feather_b = rearrange(feather_b, '(b1 b2) c -> b1 b2 c', b1=11 * 11)

    similarity1 = torch.cosine_similarity(feather_c, feather_syn_c, dim=2)
    similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=2)

    loss_c = torch.mean(1 - similarity1)
    loss_b = torch.mean(1 - similarity2)
    return loss_c,loss_b

#64的固定权重
def feather_loss_64(syn_celar,clear,syn_blur,blur, model):

    syn_clear_image_list = []
    clear_image_list = []
    syn_blur_image_list = []
    blur_image_list = []
    for i in range(13):
        for j in range(13):
            w = i * 16
            k = j * 16
            syn_clear_image_list.append(syn_celar[:, :, w:w + 64, k:k + 64])
            clear_image_list.append(clear[:, :, w:w + 64, k:k + 64])
            syn_blur_image_list.append(syn_blur[:, :, w:w + 64, k:k + 64])
            blur_image_list.append(blur[:, :, w:w + 64, k:k + 64])

    syn_clear_image = torch.cat(syn_clear_image_list, dim=0)
    clear_image = torch.cat(clear_image_list, dim=0)
    syn_blur_image = torch.cat(syn_blur_image_list, dim=0)
    blur_image = torch.cat(blur_image_list, dim=0)

    feather_syn_c, output_c = model(syn_clear_image)
    feather_c, output_cc = model(clear_image)

    feather_syn_b, output_b = model(syn_blur_image)
    feather_b, output_bb = model(blur_image)

    feather_syn_c = rearrange(feather_syn_c, '(b1 b2) c -> b1 b2 c', b1=13 * 13)
    feather_c = rearrange(feather_c, '(b1 b2) c -> b1 b2 c', b1=13 * 13)
    feather_syn_b = rearrange(feather_syn_b, '(b1 b2) c -> b1 b2 c', b1=13 * 13)
    feather_b = rearrange(feather_b, '(b1 b2) c -> b1 b2 c', b1=13 * 13)

    similarity1 = torch.cosine_similarity(feather_c, feather_syn_c, dim=2)
    similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=2)

    loss_c = torch.mean(1 - similarity1)
    loss_b = torch.mean(1 - similarity2)

    return loss_c,loss_b

#global的权重更新
def feather_loss_global_d(syn_c,all_c,syn_b,all_b,model):
    cuda_avail = torch.cuda.is_available()
    for p in model.parameters():
        p.requires_grad = False
    if cuda_avail:
        model.cuda()
    model.eval()
    loss_c_256 = 0
    loss_b_256 = 0
    for i in range(1):
        for j in range(1):
            w = i * 16
            k = j * 16
            syn_clear_image = syn_c[:, :, w:w + 256, k:k + 256]
            clear_image = all_c[:, :, w:w + 256, k:k + 256]
            feather_syn_c, output_c = model(syn_clear_image)
            feather_c, output_cc = model(clear_image)
            syn_blur_image = syn_b[:, :, w:w + 256, k:k + 256]
            blur_image = all_b[:, :, w:w + 256, k:k + 256]
            feather_syn_b, output_b = model(syn_blur_image)
            feather_b, output_bb = model(blur_image)


            similarity1 = torch.cosine_similarity(feather_c, feather_syn_c, dim=1)
            loss_c_256 = loss_c_256 + torch.mean(1 - similarity1, dim=0)
            similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=1)
            loss_b_256 = loss_b_256 + torch.mean(1 - similarity2, dim=0)

    return loss_c_256,loss_b_256

##128的权重更新
def feather_loss_128_d(syn_c,all_c,syn_b,all_b,model):
    cuda_avail = torch.cuda.is_available()
    for p in model.parameters():
        p.requires_grad = False
    if cuda_avail:
        model.cuda()
    model.eval()
    loss_c_128 = 0
    loss_b_128 = 0
    for i in range(9):
        for j in range(9):
            w = i * 16
            k = j * 16
            syn_clear_image = syn_c[:, :, w:w + 128, k:k + 128]
            clear_image = all_c[:, :, w:w + 128, k:k + 128]
            feather_syn_c, output_c = model(syn_clear_image)
            feather_c, output_cc = model(clear_image)
            syn_blur_image = syn_b[:, :, w:w + 128, k:k + 128]
            blur_image = all_b[:, :, w:w + 128, k:k + 128]
            feather_syn_b, output_b = model(syn_blur_image)
            feather_b, output_bb = model(blur_image)

            similarity1 = torch.cosine_similarity(feather_c, feather_syn_c, dim=1)
            loss_c_128 = loss_c_128 + torch.mean(1 - similarity1, dim=0)
            similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=1)
            loss_b_128 = loss_b_128 + torch.mean(1 - similarity2, dim=0)
    loss_c_128 = loss_c_128 / 81
    loss_b_128 = loss_b_128 / 81
    return loss_c_128,loss_b_128

#96的权重更新
def feather_loss_96_d(syn_c,all_c,syn_b,all_b,model):
    cuda_avail = torch.cuda.is_available()
    for p in model.parameters():
        p.requires_grad = False
    if cuda_avail:
        model.cuda()
    model.eval()
    loss_c_96 = 0
    loss_b_96 = 0
    for i in range(11):
        for j in range(11):
            w = i * 16
            k = j * 16
            syn_clear_image = syn_c[:, :, w:w + 96, k:k + 96]
            clear_image = all_c[:, :, w:w + 96, k:k + 96]
            feather_syn_c, output_c = model(syn_clear_image)
            feather_c, output_cc = model(clear_image)
            syn_blur_image = syn_b[:, :, w:w + 96, k:k + 96]
            blur_image = all_b[:, :, w:w + 96, k:k + 96]
            feather_syn_b, output_b = model(syn_blur_image)
            feather_b, output_bb = model(blur_image)

            similarity1 = torch.cosine_similarity(feather_c, feather_syn_c, dim=1)
            loss_c_96 = loss_c_96 + torch.mean(1 - similarity1, dim=0)
            similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=1)
            loss_b_96 = loss_b_96 + torch.mean(1 - similarity2, dim=0)

    loss_c_96 = loss_c_96 / 121
    loss_b_96 = loss_b_96 / 121
    return loss_c_96,loss_b_96

#64的权重更新
def feather_loss_64_d(syn_c,all_c,syn_b,all_b,model):
    cuda_avail = torch.cuda.is_available()
    for p in model.parameters():
        p.requires_grad = False
    if cuda_avail:
        model.cuda()
    model.eval()
    loss_c_64 = 0
    loss_b_64 = 0
    for i in range(13):
        for j in range(13):
            w = i * 16
            k = j * 16
            syn_clear_image = syn_c[:,:,w:w + 64, k:k + 64]
            clear_image =  all_c[:,:,w:w + 64, k:k + 64]
            feather_syn_c, output_c = model(syn_clear_image)
            feather_c, output_cc = model(clear_image)
            syn_blur_image = syn_b[:,:,w:w + 64, k:k + 64]
            blur_image =  all_b[:,:,w:w + 64, k:k + 64]
            feather_syn_b, output_b = model(syn_blur_image)
            feather_b, output_bb = model(blur_image)

            similarity1 = torch.cosine_similarity(feather_c, feather_syn_c, dim=1)
            loss_c_64 = loss_c_64 + torch.mean(1 - similarity1, dim=0)
            similarity2 = torch.cosine_similarity(feather_b, feather_syn_b, dim=1)
            loss_b_64 = loss_b_64 + torch.mean(1 - similarity2, dim=0)

        loss_c_64 = loss_c_64 / 169
        loss_b_64 = loss_b_64 / 169
        return loss_c_64, loss_b_64


#要求batchsize=4，如果后期需要改，这里再修改
def feather_loss_d5(syn_c,all_c,syn_b,all_b,model):

    loss = 0
    for i in range(1):
        for j in range(1):
            w = i * 16
            k = j * 16
            syn_clear_image = syn_c[:, :, w:w + 256, k:k + 256]
            clear_image = all_c[:, :, w:w + 256, k:k + 256]
            feather_syn_c, output_c = model(syn_clear_image)
            feather_c, output_cc = model(clear_image)
            syn_blur_image = syn_b[:, :, w:w + 256, k:k + 256]
            blur_image = all_b[:, :, w:w + 256, k:k + 256]
            feather_syn_b, output_b = model(syn_blur_image)
            feather_b, output_bb = model(blur_image)

            c_fea1 = torch.cat((feather_c, feather_syn_c), dim=0)
            c_fea1 = c_fea1.unsqueeze(0)
            c_fea = torch.cat((c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1, c_fea1), dim=0)
            # c_fea_no_diag = c_fea[~torch.eye(c_fea.shape[0],dtype=bool)].reshape((8,7,120),-1)
            c_fea_replace = c_fea.permute(1, 0, 2)
            b_fea1 = torch.cat((feather_b, feather_syn_b), dim=0)
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

    return loss

##要求batchsize=4，如果后期需要改，这里再修改
def feather_loss_d4(syn_c,all_c,syn_b,all_b, model):
    syn_clear_image_list = []
    clear_image_list = []
    syn_blur_image_list = []
    blur_image_list = []
    num_block = 9 * 9
    for i in range(9):
        for j in range(9):
            w = i * 16
            k = j * 16
            syn_clear_image_list.append(syn_c[:, :, w:w + 128, k:k + 128])
            clear_image_list.append(all_c[:, :, w:w + 128, k:k + 128])
            syn_blur_image_list.append(syn_b[:, :, w:w + 128, k:k + 128])
            blur_image_list.append(all_b[:, :, w:w + 128, k:k + 128])

    syn_clear_image = torch.cat(syn_clear_image_list, dim=0)
    clear_image = torch.cat(clear_image_list, dim=0)
    syn_blur_image = torch.cat(syn_blur_image_list, dim=0)
    blur_image = torch.cat(blur_image_list, dim=0)

    feather_syn_c, output_c = model(syn_clear_image)
    feather_c, output_cc = model(clear_image)

    feather_syn_b, output_b = model(syn_blur_image)
    feather_b, output_bb = model(blur_image)

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

    return loss

##要求batchsize=4，如果后期需要改，这里再修改
def feather_loss_d3(syn_c,all_c,syn_b,all_b,model):
    syn_clear_image_list = []
    clear_image_list = []
    syn_blur_image_list = []
    blur_image_list = []
    num_block = 11 * 11
    for i in range(11):
        for j in range(11):
            w = i * 16
            k = j * 16
            syn_clear_image_list.append(syn_c[:, :, w:w + 96, k:k + 96])
            clear_image_list.append(all_c[:, :, w:w + 96, k:k + 96])
            syn_blur_image_list.append(syn_b[:, :, w:w + 96, k:k + 96])
            blur_image_list.append(all_b[:, :, w:w + 96, k:k + 96])

    syn_clear_image = torch.cat(syn_clear_image_list, dim=0)
    clear_image = torch.cat(clear_image_list, dim=0)
    syn_blur_image = torch.cat(syn_blur_image_list, dim=0)
    blur_image = torch.cat(blur_image_list, dim=0)

    feather_syn_c, output_c = model(syn_clear_image)
    feather_c, output_cc = model(clear_image)

    feather_syn_b, output_b = model(syn_blur_image)
    feather_b, output_bb = model(blur_image)

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

    return loss

#要求batchsize=4，如果后期需要改，这里再修改
def feather_loss_d2(syn_c,all_c,syn_b,all_b,model):
    syn_clear_image_list = []
    clear_image_list = []
    syn_blur_image_list = []
    blur_image_list = []
    num_block = 13 * 13
    for i in range(13):
        for j in range(13):
            w = i * 16
            k = j * 16
            syn_clear_image_list.append(syn_c[:, :, w:w + 64, k:k + 64])
            clear_image_list.append(all_c[:, :, w:w + 64, k:k + 64])
            syn_blur_image_list.append(syn_b[:, :, w:w + 64, k:k + 64])
            blur_image_list.append(all_b[:, :, w:w + 64, k:k + 64])

    syn_clear_image = torch.cat(syn_clear_image_list, dim=0)
    clear_image = torch.cat(clear_image_list, dim=0)
    syn_blur_image = torch.cat(syn_blur_image_list, dim=0)
    blur_image = torch.cat(blur_image_list, dim=0)

    feather_syn_c, output_c = model(syn_clear_image)
    feather_c, output_cc = model(clear_image)

    feather_syn_b, output_b = model(syn_blur_image)
    feather_b, output_bb = model(blur_image)

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

    return loss