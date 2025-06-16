import time
from functools import partial
from typing import Callable, Tuple, List
import argparse

import numpy as np
import torch
from math import ceil
from torch import Tensor
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets.cifar import CIFAR100
from cl_dataset_tools import NCProtocol, NCProtocolIterator, TransformationDataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
from DatasetModule import MyData
from cl_strategies import icarl_accuracy_measure, icarl_cifar100_augment_data
from models import make_icarl_net
from cl_metrics_tools import get_accuracy
from models.icarl_net import IcarlNet, initialize_icarl_net
from models.resnet import ResNet18Cut
from utils import get_dataset_per_pixel_mean, make_theano_training_function, make_theano_validation_function, \
    make_theano_feature_extraction_function, make_theano_inference_function, make_batch_one_hot, retrieval_performances,make_theano_training_function_with_features
from PIL import Image

from utils.correlation_loss import  Similarity_preserving

import pandas as pd
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# Line 31: Load the dataset
# Asserting nb_val == 0 equals to full cifar100
# That is, we only declare transformations here
# Notes: dstack and reshape already done inside CIFAR100 class
# Mean is calculated on already scaled (by /255) images

# transform = transforms.Compose([
#     transforms.ToTensor(),  # ToTensor scales from [0, 255] to [0, 1.0]
# ])

# per_pixel_mean = get_dataset_per_pixel_mean(create("cub", transform = transform))
# def transform1(x):
#     return x - per_pixel_mean

class CustomLoss(torch.nn.Module):
    def __init__(self, loss1: torch.nn.Module = BCELoss(), loss2: torch.nn.Module = Similarity_preserving()):
        super(CustomLoss, self).__init__()
        self.loss1  = loss1
        self.loss2  = loss2
    def forward(self, inputs, targets, features, features2, beta_0=1):
        # Apply sigmoid to inputs
        # Calculate binary cross-entropy loss
        # print("input min", inputs.min())
        # print("input max", inputs.max())
        # print("target values count", targets.unique(return_counts=True))
        loss1 = self.loss1(inputs, targets)

        loss2 = self.loss2(features, features2)
        return beta_0*loss2 + loss1



class CovertBGR(object):
    def __init__(self):
        pass

    def __call__(self, img):
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
        return img

def main():

    print(f"Using device {device}")

    # This script tries to reprodice results of official iCaRL code
    # https://github.com/srebuffi/iCaRL/blob/master/iCaRL-TheanoLasagne/main_cifar_100_theano.py

    #Adding argparse for let user choose the number of epochs
    parser = argparse.ArgumentParser(description="iCaRL Training Script")
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
    args = parser.parse_args()


    ######### Modifiable Settings ##########
    batch_size = 80            # Batch size
    n          = 5              # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
    # nb_val     = 0            # Validation samples per class
    nb_cl      = 10             # Classes per group
    nb_protos  = 20             # Number of prototypes per class at the end: total protoset memory/ total number of classes
    epochs     = args.epochs    # Total number of epochs
    lr_old     = 2.             # Initial learning rate
    lr_strat   = [49, 63]       # Epochs where learning rate gets decreased
    lr_factor  = 5.             # Learning rate decrease factor
    wght_decay = 0.00001        # Weight Decay
    nb_runs    = 1              # Number of runs (random ordering of classes at each run)
    beta_0 = 1 # Beta parameter for the loss
    torch.manual_seed(2000)     # Fix the random seed
    ratio = 0.16          # Ratio of the random crop (0.2 means 20% of the image size)
    origin_width = 256   # Original width of the image
    width = 227  # Width after random crop

    ########################################

    fixed_class_order = [137, 41, 112, 7, 122, 186, 93, 101, 10, 44, 23, 195, 31, 173, 2, 59, 83, 123, 1, 99,
    191, 24, 110, 180, 70, 19, 56, 26, 38, 33, 11, 148, 100, 115, 124, 151, 109, 104, 25, 49,
    199, 154, 164, 77, 14, 55, 194, 176, 66, 150, 35, 128, 88, 132, 179, 183, 156, 8, 13, 54,
    153, 36, 145, 18, 134, 102, 143, 192, 65, 175, 152, 119, 3, 198, 43, 178, 116, 125, 144, 105,
    140, 69, 129, 135, 157, 139, 126, 164, 6, 160, 120, 184, 167, 147, 138, 197, 159, 12, 130, 189,
    22, 142, 17, 171, 5, 174, 61, 185, 141, 0, 20, 40, 27, 4, 42, 168, 162, 107, 127, 182, 111, 146,
    34, 117, 21, 149, 118, 9, 106, 161, 163, 72, 170, 50, 16, 32, 166, 169, 165, 15, 190, 48, 46, 39,
    71, 136, 58, 53, 67, 47, 157, 28, 108, 114, 78, 76, 60, 81, 52, 177, 63, 68, 73, 133, 94, 187,
    188, 29, 51, 45, 113, 98, 95, 87, 196, 57, 84, 85, 79, 80, 75, 74, 82, 86, 181, 58, 90, 91, 92,
    96, 97, 103, 172, 155, 193, 30, 37, 62]

    # fixed_class_order = None


    # https://github.com/srebuffi/iCaRL/blob/90ac1be39c9e055d9dd2fa1b679c0cfb8cf7335a/iCaRL-TheanoLasagne/utils_cifar100.py#L146

    std_value = 1.0 / 255.0
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std= [std_value, std_value, std_value])
    transform = \
    transforms.Compose([
                CovertBGR(),
                transforms.Resize((origin_width)),
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
               ])

    transform_test = \
    transforms.Compose([
                    CovertBGR(),
                    transforms.Resize((origin_width)),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])
    
    transform_prototypes = \
    transforms.Compose([
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.RandomHorizontalFlip()
                ])



    # Line 43: Initialization
    dictionary_size = 500
    top1_acc_list_cumul = torch.zeros(200//nb_cl, 3, nb_runs)
    top1_acc_list_ori = torch.zeros(200//nb_cl, 3, nb_runs)
    top1_acc_list_curr = torch.zeros(200//nb_cl, 3, nb_runs)
    top1_acc_list_cumul = torch.zeros(200//nb_cl, 3, nb_runs)        
    top1_acc_list_curr_train = torch.zeros(200//nb_cl, 3, nb_runs)
    top1_acc_list_cumul_train = torch.zeros(200//nb_cl, 3, nb_runs)



    map_whole = torch.zeros(200//nb_cl, 2)
    metrics = torch.zeros(200//nb_cl, 6)
    losses = torch.zeros(200//nb_cl, epochs, 2)
    time_list = torch.zeros(200//nb_cl, epochs)

    # Line 48: # Launch the different runs
    # Skipped as this script will only manage singe runs

    # Lines 51, 52, 54 already managed in NCProtocol

    protocol = NCProtocol(MyData('./data/CUB_200_2011', label_txt='data/CUB_200_2011/train.txt', transform=transform),
                          MyData('./data/CUB_200_2011', label_txt='data/CUB_200_2011/test.txt', transform=transform_test),
                          n_tasks=200//nb_cl, shuffle=True, seed=None, fixed_class_order=fixed_class_order)

    model: IcarlNet = make_icarl_net(num_classes=200)#ResNet18Cut()# 
    model.apply(initialize_icarl_net)

    model = model.to(device)

    criterion = CustomLoss()  # Line 66-67

    # Line 74, 75
    # Note: sh_lr is a theano "shared"
    sh_lr = lr_old

    # noinspection PyTypeChecker
    val_fn: Callable[[Tensor, Tensor],
                     Tuple[Tensor, Tensor, Tensor]] = partial(make_theano_validation_function, model,
                                                              CustomLoss(), 'feature_extractor',
                                                              device=device)

    # noinspection PyTypeChecker
    function_map: Callable[[Tensor], Tensor] = partial(make_theano_feature_extraction_function, model,
                                                       'feature_extractor', device=device, batch_size=batch_size)

    # Lines 90-97: Initialization of the variables for this run

    x_protoset_cumuls: List[Tensor] = []
    y_protoset_cumuls: List[Tensor] = []
    alpha_dr_herding = torch.zeros((200 // nb_cl, dictionary_size, nb_cl), dtype=torch.float)

    # Lines 101-103: already managed by NCProtocol/NCProtocolIterator

    train_dataset: Dataset
    task_info: NCProtocolIterator

    func_pred: Callable[[Tensor], Tensor]
    # func_pred_feat: Callable[[Tensor], Tensor] # Unused

    for task_idx, (train_dataset, task_info) in enumerate(protocol):
        print('Classes in this batch:', task_info.classes_in_this_task)

        # Lines 107, 108: Save data results at each increment
        # torch.save(top1_acc_list_cumul, 'top1_acc_list_cumul_icarl_cl' + str(nb_cl))
        # torch.save(top1_acc_list_ori, 'top1_acc_list_ori_icarl_cl' + str(nb_cl))
        # torch.save(map_whole, 'map_whole' + str(nb_cl))

        # Note: lines 111-125 already managed in NCProtocol/NCProtocolIterator

        # Lines 128-135: Add the stored exemplars to the training data
        # Note: X_valid_ori and Y_valid_ori already managed in NCProtocol/NCProtocolIterator
        if task_idx != 0:
            protoset = TransformationDataset(TensorDataset(torch.cat(x_protoset_cumuls), torch.cat(y_protoset_cumuls)),
                                             transform=transform_prototypes, target_transform=None)
            train_dataset = ConcatDataset((train_dataset, protoset))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

if __name__ == "__main__":
    main()