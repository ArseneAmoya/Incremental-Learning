import torch
from torchvision.datasets.cifar import CIFAR100
from cl_dataset_tools import NCProtocol, NCProtocolIterator
from torchvision.models.resnet import resnet18
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import BCELoss
from models import ResNet18Cut
import time
from argparse import ArgumentParser
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from utils import make_batch_one_hot
import pandas as pd
from torch.utils.data import ConcatDataset
from typing import List
from models.icarl_net import initialize_icarl_net
from models.icarl_net import make_icarl_net
from utils import get_dataset_per_pixel_mean
from utils import make_theano_training_function
from cl_strategies import icarl_cifar100_augment_data
from PIL import Image
import argparse
from DatasetModule import MyData
from models.icarl_net import IcarlNet
from typing import Callable, Tuple, List
from torch import Tensor

from utils import retrieval_performances

from cl_metrics_tools import get_accuracy

from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

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
    n_epochs     = args.epochs    # Total number of epochs
    lr_old     = 2.             # Initial learning rate
    lr_strat   = [49, 63]       # Epochs where learning rate gets decreased
    lr_factor  = 5.             # Learning rate decrease factor
    wght_decay = 0.00001        # Weight Decay
    nb_runs    = 1              # Number of runs (random ordering of classes at each run)
    beta_0 = 1 # Beta parameter for the loss
    torch.manual_seed(2000)     # Fix the random seed
    ratio = 0.16          # Ratio of the random crop (0.2 means 20% of the image size)
    origin_width = 256   # Original width of the image
    top_k_accuracies = [1, 5]   # Top-k accuracies to compute

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
    

    # Line 43: Initialization



    map_list = torch.zeros(200//nb_cl, 2)
    metrics = torch.zeros(200//nb_cl, 10)
    losses = torch.zeros(200//nb_cl, n_epochs, 2)
    time_list = torch.zeros(200//nb_cl, n_epochs)

    tasks = 200 // nb_cl  # Number of tasks (200 classes, nb_cl classes per task)
    # Line 48: # Launch the different runs
    # Skipped as this script will only manage singe runs

    # Lines 51, 52, 54 already managed in NCProtocol

    protocol = NCProtocol(MyData('./data/CUB_200_2011', label_txt='data/CUB_200_2011/train.txt', transform=transform),
                          MyData('./data/CUB_200_2011', label_txt='data/CUB_200_2011/test.txt', transform=transform_test),
                          n_tasks=200//nb_cl, shuffle=True, seed=None, fixed_class_order=fixed_class_order)

    model: IcarlNet = make_icarl_net(num_classes=200)#ResNet18Cut()# 
    model.apply(initialize_icarl_net)

    model = model.to(device)

    criterion = BCELoss()  # Line 66-67

    # Line 74, 75
    # Note: sh_lr is a theano "shared"
    sh_lr = lr_old
    # Lines 101-103: already managed by NCProtocol/NCProtocolIterator

    train_dataset: Dataset
    task_info: NCProtocolIterator

    func_pred: Callable[[Tensor], Tensor]
    # func_pred_feat: Callable[[Tensor], Tensor] # Unused
    cumulative_datasets: List = []

    for task_idx, (train_ds, task_info) in enumerate(protocol):
        print()
        print('-------------------------------------------------------------------------------')
        print('Task', task_idx, 'started')
        print('Classes in this batch:', task_info.classes_in_this_task)

        cumulative_datasets.append(train_ds)
        train_dataset = ConcatDataset(cumulative_datasets)
        train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=8)

        optimizer = torch.optim.SGD(model.parameters(), lr=sh_lr, weight_decay=wght_decay, momentum=0.9)
        train_fn = partial(make_theano_training_function, model, criterion, optimizer, device=device)
        scheduler = MultiStepLR(optimizer, lr_strat, gamma=1.0/lr_factor)



        for epoch in range(n_epochs):
            start_time = time.time()
            epoch_loss = 0
                  # Sets model in train mode
            model.train()
            for patterns, labels in train_loader:

                targets = make_batch_one_hot(labels, 200)

                # Clear grad
                #optimizer.zero_grad()

                # Send data to device
                patterns = patterns.to(device)
                targets = targets.to(device)

                # Forward
                #output = model(patterns)

                # Loss
                #loss = criterion(output, targets)
                epoch_loss += train_fn(patterns, targets) #loss.item()

                # Update step            
            scheduler.step()
            epoch_time = time.time() - start_time
            time_list[task_idx, epoch] = epoch_time
            losses[task_idx, epoch, 0] = epoch_loss / len(train_loader)
            # ---- Validation ----
            model.eval()
            with torch.no_grad():
                val_accuracies, val_loss, _, _ = get_accuracy(model, task_info.get_current_test_set(), device=device,
                                                 required_top_k=[1, 5], return_detailed_outputs=False,
                                                 criterion=BCELoss(), make_one_hot=True, n_classes=200,
                                                 batch_size=80, shuffle=False, num_workers=8)
                losses[task_idx, epoch, 1] = val_loss
            print(f'Epoch {epoch} train loss: {epoch_loss/len(train_loader):.5f} validation loss {val_loss:.5f} training time {epoch_time:.3f} seconds')    
        print('Task', task_idx, 'ended')

        top_train_accuracies, _, _, _ = get_accuracy(model,
                                                  task_info.swap_transformations().get_cumulative_training_set(),
                                                  device=device, required_top_k=top_k_accuracies, batch_size=80)

        top_train_accuracies_current, _, _, _ = get_accuracy(model,
                                                  task_info.swap_transformations().get_current_training_set(),
                                                  device=device, required_top_k=top_k_accuracies, batch_size=80)
        top_test_accuracies, _, _, _ = get_accuracy(model, task_info.get_cumulative_test_set(), device=device,
                                                 required_top_k=top_k_accuracies, batch_size=80)
        
        top_test_accuracies_current, _, _, _ = get_accuracy(model, task_info.get_current_test_set(), device=device,
                                                    required_top_k=top_k_accuracies, batch_size=80)
        map_list = retrieval_performances(task_info.get_cumulative_test_set(), model, map_list, task_idx, batch_size=80, current_classes=task_info.classes_in_this_task)

        for top_k_idx, top_k_acc in enumerate(top_k_accuracies):
            print('Top', top_k_acc, 'train current set accuracy {:.4f}'.format(top_train_accuracies_current[top_k_idx].item()))
            print('Top', top_k_acc, 'train cumul set accuracy {:.4f}'.format(top_train_accuracies[top_k_idx].item()))
            print('Top', top_k_acc, 'test current set accuracy {:.4f}'.format(top_test_accuracies_current[top_k_idx].item()))
            print('Top', top_k_acc, 'test cumul set accuracy {:.4f}'.format(top_test_accuracies[top_k_idx].item()))
            print('\n')

            metrics[task_idx, top_k_idx*4] = top_train_accuracies_current[top_k_idx].item()
            metrics[task_idx, top_k_idx*4+1] = top_train_accuracies[top_k_idx].item()
            metrics[task_idx, top_k_idx*4+2] = top_test_accuracies_current[top_k_idx].item()
            metrics[task_idx, top_k_idx*4+3] = top_test_accuracies[top_k_idx].item()

        metrics[task_idx, 8] = map_list[task_idx, 0].item()
        metrics[task_idx, 9] = map_list[task_idx, 1].item()

        torch.save(model.state_dict(), 'net_naive_2'+str(task_idx+1)+'_of_'+str(task_idx))
        torch.save(model.feature_extractor.state_dict(), 'intermed_naive_2'+str(task_idx+1)+'_of_'+str(task_idx))

        metrics_df = pd.DataFrame(metrics.numpy(), columns= ['top1_train_current', 'top1_train_cumul', 'top1_test_current', 'top1_test_cumul',
                                                                        'top5_train_current', 'top5_train_cumul', 'top5_test_current', 'top5_test_cumul',
                                                                        'map_cumulative', 'map_current'])
        metrics_df.to_csv('metrics.csv', index=False)

        time_df = pd.DataFrame(time_list.numpy(), columns=None)
        losses_df = pd.DataFrame(losses.view(tasks*n_epochs, 2).numpy(), columns=['loss', 'val_loss'])
        #concat_df = pd.concat([time_df, losses_df], axis=1)
        losses_df.to_csv('losses.csv', index=False)
        time_df.to_csv('time.csv', index=False)



if __name__ == '__main__':
    main()
