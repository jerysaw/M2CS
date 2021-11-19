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
from models.image_classification_net_64 import img_clif_net as img_clif_net_64
from models.image_classification_net_96 import img_clif_net as img_clif_net_96
from models.image_classification_net_128 import img_clif_net as img_clif_net_128
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
#load the args

model = ResNet()
# path = './parameters/model_resnet_256_256.pth'
# model.load_state_dict(torch.load(path))
model.train()
model.cuda()

optimizer = Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

path_clear = './dataset/train_data/dis_images/FC_train'
path_blur = './dataset/train_data/dis_images/FB_train'

dataloder_C1 = Get_dataloader(path_clear, 4)
dataloder_C2 = Get_dataloader(path_clear, 4)

dataloder_B1 = Get_dataloader(path_blur, 4)
dataloder_B2 = Get_dataloader(path_blur, 4)

loader_C1 = iter(dataloder_C1)
loader_C2 = iter(dataloder_C2)

loader_B1 = iter(dataloder_B1)
loader_B2 = iter(dataloder_B2)

loss_print = 0
for i in range(100_000):
    try:
        img_C1 = next(loader_C1).cuda()
        img_C2 = next(loader_C2).cuda()
        img_B1 = next(loader_B1).cuda()
        img_B2 = next(loader_B2).cuda()
    except (OSError, StopIteration):
        loader_C1 = iter(dataloder_C1)
        loader_C2 = iter(dataloder_C2)
        loader_B1 = iter(dataloder_B1)
        loader_B2 = iter(dataloder_B2)

        img_C1 = next(loader_C1).cuda()
        img_C2 = next(loader_C2).cuda()
        img_B1 = next(loader_B1).cuda()
        img_B2 = next(loader_B2).cuda()

    optimizer.zero_grad()
    vector_C1, _ = model(img_C1)
    vector_C2, _ = model(img_C2)
    vector_B1, _ = model(img_B1)
    vector_B2, _ = model(img_B2)

    loss_sim = 2 - torch.mean(
        torch.cosine_similarity(vector_C1, vector_C2, dim=1) + torch.cosine_similarity(vector_B1, vector_B2, dim=1))
    loss_unsim =2 + torch.mean(
        torch.cosine_similarity(vector_C1, vector_B2, dim=1) + torch.cosine_similarity(vector_B1, vector_C2, dim=1))

    loss = loss_sim + loss_unsim

    print(loss)
    loss.backward()
    optimizer.step()