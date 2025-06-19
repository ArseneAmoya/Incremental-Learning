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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    dictionary_size = 30
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

    criterion = BCELoss()  # Line 66-67

    # Line 74, 75
    # Note: sh_lr is a theano "shared"
    sh_lr = lr_old

    # noinspection PyTypeChecker
    val_fn: Callable[[Tensor, Tensor],
                     Tuple[Tensor, Tensor, Tensor]] = partial(make_theano_validation_function, model,
                                                              BCELoss(), 'feature_extractor',
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        # Line 137: # Launch the training loop
        # From lines: 69, 70, 76, 77
        # Note optimizer == train_fn
        # weight_decay == l2_penalty
        optimizer = torch.optim.SGD(model.parameters(), lr=sh_lr, weight_decay=wght_decay, momentum=0.9)
        train_fn = partial(make_theano_training_function, model, criterion, optimizer, device=device)
        scheduler = MultiStepLR(optimizer, lr_strat, gamma=1.0/lr_factor)

        print("\n")

        # Added (not found in original code): validation accuracy before first epoch
        acc_result, val_err, _, _ = get_accuracy(model, task_info.get_current_test_set(), device=device,
                                                 required_top_k=[1, 5], return_detailed_outputs=False,
                                                 criterion=BCELoss(), make_one_hot=True, n_classes=200,
                                                 batch_size=batch_size, shuffle=False, num_workers=8)
        print("Before first epoch")
        print("  validation loss:\t\t{:.6f}".format(val_err))  # Note: already averaged
        print("  top 1 accuracy:\t\t{:.2f} %".format(acc_result[0].item() * 100))
        print("  top 5 accuracy:\t\t{:.2f} %".format(acc_result[1].item() * 100))
        # End of added code

        print('Batch of classes number {0} arrives ...'.format(task_idx + 1))

        # Sets model in train mode
        model.train()
        for epoch in range(epochs):
            # Note: already shuffled (line 143-146)

            # Lines 148-150
            train_err: float = 0
            #train_batches: int = 0
            start_time: float = time.time()

            patterns: Tensor
            labels: Tensor
            for patterns, labels in train_loader:  # Line 151
                # Lines 153-154
                targets = make_batch_one_hot(labels, 200)

                #old_train = train_err  # Line 155

                targets = targets.to(device)
                patterns = patterns.to(device)

                if task_idx == 0:   # Line 156
                    train_err += train_fn(patterns, targets)  # Line 157

                # Lines 160-163: Distillation
                if task_idx > 0:
                    prediction_old = func_pred(patterns)
                    targets[:, task_info.prev_classes] = prediction_old[:, task_info.prev_classes]
                    train_err += train_fn(patterns, targets)

                # if (train_batches % 100) == 1:
                #     print(train_err - old_train)

                #train_batches += 1
            scheduler.step()
            epoch_time = time.time() - start_time

            # Lines 171-186: And a full pass over the validation data:
            acc_result, val_err, _, _ = get_accuracy(model, task_info.get_current_test_set(),  device=device,
                                                     required_top_k=[1, 5], return_detailed_outputs=False,
                                                     criterion=BCELoss(), make_one_hot=True, n_classes=200,
                                                     batch_size=batch_size, shuffle=False, num_workers=8)
            

            # Lines 188-202: Then we print the results for this epoch:
            print("Batch of classes {} out of {} batches".format(
                task_idx + 1, 200 // nb_cl))
            time_list[task_idx, epoch] = epoch_time
            losses[task_idx, epoch, 0] = train_err / len(train_loader)
            losses[task_idx, epoch, 1] = val_err
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1,
                epochs,
                epoch_time))
            print("  training loss:\t\t{:.6f}".format(train_err / len(train_loader)))
            print("  validation loss:\t\t{:.6f}".format(val_err))  # Note: already averaged
            print("  top 1 accuracy:\t\t{:.2f} %".format(
                acc_result[0].item() * 100))
            print("  top 5 accuracy:\t\t{:.2f} %".format(
                acc_result[1].item() * 100))
            # adjust learning rate

        # Lines 205-213: Duplicate current network to distillate info
        if task_idx == 0:
            model2 = make_icarl_net(200, n=n) #ResNet18Cut()#
            model2 = model2.to(device)
            # noinspection PyTypeChecker
            func_pred = partial(make_theano_inference_function, model2, device=device)

            # Note: func_pred_feat is unused
            # func_pred_feat = partial(make_theano_feature_extraction_function, model=model2,
            #                          feature_extraction_layer='feature_extractor')

        model2.load_state_dict(model.state_dict())

        # Lines 216, 217: Save the network
        torch.save(model.state_dict(), 'net_incr'+str(task_idx+1)+'_of_'+str(100//nb_cl))
        torch.save(model.feature_extractor.state_dict(), 'intermed_incr'+str(task_idx+1)+'_of_'+str(100//nb_cl))

        # Lines 220-242: Exemplars
        nb_protos_cl = int(ceil(nb_protos * 200. / nb_cl / (task_idx + 1)))

        # Herding
        print('Updating exemplar set...')
        for iter_dico in range(nb_cl):
            # Possible exemplars in the feature space and projected on the L2 sphere
            prototypes_for_this_class, _ = task_info.swap_transformations() \
                .get_current_training_set()[iter_dico*dictionary_size:(iter_dico+1)*dictionary_size]

            mapped_prototypes: Tensor = function_map(prototypes_for_this_class)
            D: Tensor = mapped_prototypes.T
            D = D / torch.norm(D, dim=0)

            # Herding procedure : ranking of the potential exemplars
            mu = torch.mean(D, dim=1)
            alpha_dr_herding[task_idx, :, iter_dico] = alpha_dr_herding[task_idx, :, iter_dico] * 0
            w_t = mu
            iter_herding = 0
            iter_herding_eff = 0
            while not (torch.sum(alpha_dr_herding[task_idx, :, iter_dico] != 0) ==
                       min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
                tmp_t = torch.mm(w_t.unsqueeze(0), D)
                ind_max = torch.argmax(tmp_t)
                iter_herding_eff += 1
                if alpha_dr_herding[task_idx, ind_max, iter_dico] == 0:
                    alpha_dr_herding[task_idx, ind_max, iter_dico] = 1 + iter_herding
                    iter_herding += 1
                w_t = w_t + mu - D[:, ind_max]

        # Lines 244-246: Prepare the protoset
        x_protoset_cumuls: List[Tensor] = []
        y_protoset_cumuls: List[Tensor] = []

        # Lines 249-276: Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
        print('Computing mean-of_exemplars and theoretical mean...')
        class_means = torch.zeros((64, 200, 2), dtype=torch.float)
        for iteration2 in range(task_idx + 1):
            for iter_dico in range(nb_cl):
                prototypes_for_this_class: Tensor
                current_cl = task_info.classes_seen_so_far[list(
                    range(iteration2 * nb_cl, (iteration2 + 1) * nb_cl))]
                current_class = current_cl[iter_dico].item()

                prototypes_for_this_class, _ = task_info.swap_transformations().get_task_training_set(iteration2)[
                                            iter_dico * dictionary_size:(iter_dico + 1)*dictionary_size]

                # Collect data in the feature space for each class
                mapped_prototypes: Tensor = function_map(prototypes_for_this_class)
                D: Tensor = mapped_prototypes.T
                D = D / torch.norm(D, dim=0)

                # Flipped version also
                # PyTorch doesn't support ::-1 yet
                # And with "yet" I mean: PyTorch will NEVER support ::-1
                # See: https://github.com/pytorch/pytorch/issues/229 (<-- year 2016!)
                # Also: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
                mapped_prototypes2: Tensor = function_map(torch.from_numpy(
                    prototypes_for_this_class.numpy()[:, :, :, ::-1].copy()))
                D2: Tensor = mapped_prototypes2.T
                D2 = D2 / torch.norm(D2, dim=0)

                # iCaRL
                alph = alpha_dr_herding[iteration2, :, iter_dico]
                alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.

                # Adds selected replay patterns
                x_protoset_cumuls.append(prototypes_for_this_class[torch.where(alph == 1)[0]])
                # Appends labels of replay patterns -> Tensor([current_class, current_class, current_class, ...])
                y_protoset_cumuls.append(current_class * torch.ones(len(torch.where(alph == 1)[0])))
                alph = alph / torch.sum(alph)
                class_means[:, current_cl[iter_dico], 0] = (torch.mm(D, alph.unsqueeze(1)).squeeze(1) +
                                                            torch.mm(D2, alph.unsqueeze(1)).squeeze(1)) / 2
                class_means[:, current_cl[iter_dico], 0] /= torch.norm(class_means[:, current_cl[iter_dico], 0])

                # Normal NCM
                alph = torch.ones(dictionary_size) / dictionary_size
                class_means[:, current_cl[iter_dico], 1] = (torch.mm(D, alph.unsqueeze(1)).squeeze(1) +
                                                            torch.mm(D2, alph.unsqueeze(1)).squeeze(1)) / 2

                class_means[:, current_cl[iter_dico], 1] /= torch.norm(class_means[:, current_cl[iter_dico], 1])

        torch.save(class_means, 'cl_means')  # Line 278

        # Calculate validation error of model on the first nb_cl classes:
        print('Computing accuracy on test sets...')
        top1_acc_list_curr = icarl_accuracy_measure(task_info.get_current_test_set(), class_means, val_fn,
                                                   top1_acc_list_curr, task_idx, 0, 'Current test set',
                                                   make_one_hot=True, n_classes=200,
                                                   batch_size=batch_size, num_workers=8)

        top1_acc_list_cumul = icarl_accuracy_measure(task_info.get_cumulative_test_set(), class_means, val_fn,
                                                     top1_acc_list_cumul, task_idx, 0, 'cumul of test set',
                                                     make_one_hot=True, n_classes=200,
                                                     batch_size=batch_size, num_workers=8)
        
        top1_acc_list_curr_train = icarl_accuracy_measure(task_info.swap_transformations().get_current_training_set(), class_means, val_fn,
                                                   top1_acc_list_curr_train, task_idx, 0, 'Current train set',
                                                   make_one_hot=True, n_classes=200,
                                                   batch_size=batch_size, num_workers=8)

        top1_acc_list_cumul_train = icarl_accuracy_measure(task_info.swap_transformations().get_cumulative_training_set(), class_means, val_fn,
                                                     top1_acc_list_cumul_train, task_idx, 0, 'cumul of Train set',
                                                     make_one_hot=True, n_classes=200,
                                                     batch_size=batch_size, num_workers=8)
        metrics[task_idx, 2] = top1_acc_list_curr[task_idx, 0]

        metrics[task_idx, 3] = top1_acc_list_cumul[task_idx, 0]
        
        metrics[task_idx, 0] = top1_acc_list_curr_train[task_idx, 0]

        metrics[task_idx, 1] = top1_acc_list_cumul_train[task_idx, 0]


        map_whole = retrieval_performances(task_info.get_cumulative_test_set(), model, map_whole, task_idx, current_classes=task_info.classes_in_this_task,
                                           batch_size=batch_size)
        metrics[task_idx, 4] = map_whole[task_idx, 0]
        #metrics[task_idx, 1] = top1_acc_list_ori[task_idx, 0]
        metrics[task_idx, 5] = map_whole[task_idx, 1]
    

    
    # Final save of the data

    # torch.save(top1_acc_list_cumul, 'top1_acc_list_cumul_icarl_cl' + str(nb_cl))
    # torch.save(top1_acc_list_ori, 'top1_acc_list_ori_icarl_cl' + str(nb_cl))
    # torch.save(map_whole, 'map_list_cumul_icarl_cl' + str(nb_cl))
    metrics_df = pd.DataFrame(metrics.numpy(), columns= ['top1_train_current', 'top1_train_cumul', 'top1_test_current', 'top1_test_cumul',
                                                        'map_cumul','map_current'])
    metrics_df.to_csv('metrics.csv', index=False)

    time_df = pd.DataFrame(time_list.numpy(), columns=None)
    losses_df = pd.DataFrame(losses.view(200//nb_cl*epochs, 2).numpy(), columns=['loss', 'val_loss'])
    #concat_df = pd.concat([time_df, losses_df], axis=1)
    losses_df.to_csv('losses.csv', index=False)
    time_df.to_csv('time.csv', index=False)

if __name__ == '__main__':
    main()
