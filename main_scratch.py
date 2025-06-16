import time
from functools import partial
from typing import Callable, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets.cifar import CIFAR100
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms

from cl_dataset_tools import NCProtocol
from cl_strategies import icarl_cifar100_augment_data
from cl_metrics_tools import get_accuracy
from models import make_icarl_net
from models.icarl_net import IcarlNet, initialize_icarl_net
from utils import get_dataset_per_pixel_mean, make_theano_training_function, make_batch_one_hot
from tqdm import tqdm

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Using device {device}")

# Data transforms and mean subtraction
base_transform = transforms.Compose([transforms.ToTensor()])
per_pixel_mean = get_dataset_per_pixel_mean(
    CIFAR100('./data/cifar100', train=True, download=True, transform=base_transform)
)
def transform1(x: Tensor) -> Tensor:
    return x - per_pixel_mean

transform = transforms.Compose([
    transforms.ToTensor(),
    transform1,
    icarl_cifar100_augment_data,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transform1,
])

def main():
    ######### Modifiable Settings ##########
    batch_size = 128            # Batch size
    n          = 5              # ResNet depth parameter (unused here)
    nb_cl      = 10             # Classes per session
    epochs     = 70             # Epochs per session
    lr_old     = 2.0            # Initial learning rate
    lr_strat   = [49, 63]       # LR decay epochs
    lr_factor  = 5.0            # LR decay factor
    wght_decay = 1e-5           # Weight decay
    nb_runs    = 1              # Number of runs
    torch.manual_seed(1993)     # Seed for reproducibility
    ########################################

    fixed_class_order = [
        87,  0, 52, 58, 44, 91, 68, 97, 51, 15,
        94, 92, 10, 72, 49, 78, 61, 14,  8, 86,
        84, 96, 18, 24, 32, 45, 88, 11,  4, 67,
        69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
        17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
         1, 28,  6, 46, 62, 82, 53,  9, 31, 75,
        38, 63, 33, 74, 27, 22, 36,  3, 16, 21,
        60, 19, 70, 90, 89, 43,  5, 42, 65, 76,
        40, 30, 23, 85,  2, 95, 56, 48, 71, 64,
        98, 13, 99,  7, 34, 55, 54, 26, 35, 39
    ]

    # Metrics storage
    top1_acc_list_cumul = torch.zeros(100 // nb_cl, 3, nb_runs)
    top1_acc_list_ori   = torch.zeros(100 // nb_cl, 3, nb_runs)
    map_whole = torch.zeros(100//nb_cl, nb_runs)

    # Prepare continual-learning protocol
    protocol = NCProtocol(
        CIFAR100('./data/cifar100', train=True, download=True, transform=transform),
        CIFAR100('./data/cifar100', train=False, download=True, transform=transform_test),
        n_tasks=100 // nb_cl,
        shuffle=True,
        fixed_class_order=fixed_class_order
    )

    # Model initialization
    model: IcarlNet = make_icarl_net(num_classes=100)
    model.apply(initialize_icarl_net)
    model = model.to(device)
    criterion = BCELoss()
    sh_lr = lr_old

    # Keep track of seen training data
    cumulative_datasets: List = []

    for task_idx, (train_ds, task_info) in enumerate(protocol):
        print('Classes in this batch:', task_info.classes_in_this_task)

        # Save metrics at each increment
        torch.save(top1_acc_list_cumul, 'top1_acc_list_cumul_icarl_cl' + str(nb_cl))
        torch.save(top1_acc_list_ori,   'top1_acc_list_ori_icarl_cl'   + str(nb_cl))
        torch.save(map_whole, 'map_whole' + str(nb_cl))

        # Build cumulative training set
        cumulative_datasets.append(train_ds)
        train_dataset = ConcatDataset(cumulative_datasets)
        train_loader  = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=0)
        print("Cumulative dataset size:", len(train_dataset))
        print("Current task dataset size:", len(train_ds))
        print("Current task dataloader size:", len(train_loader))

        # Optimizer, scheduler and training function
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=sh_lr,
                                    weight_decay=wght_decay,
                                    momentum=0.9)
        scheduler = MultiStepLR(optimizer,
                                lr_strat,
                                gamma=1.0 / lr_factor)
        train_fn = partial(make_theano_training_function,
                           model,
                           criterion,
                           optimizer,
                           device=device)

        print("\n")
        # Validation before training
        acc_result, val_err, _, _ = get_accuracy(
            model,
            task_info.get_current_test_set(),
            device=device,
            required_top_k=[1, 5],
            return_detailed_outputs=False,
            criterion=BCELoss(),
            make_one_hot=True,
            n_classes=100,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )
        print("Before first epoch")
        print("  validation loss:		{:.6f}".format(val_err))
        print("  top 1 accuracy:		{:.2f} %".format(acc_result[0].item() * 100))
        print("  top 5 accuracy:		{:.2f} %".format(acc_result[1].item() * 100))

        print('Batch of classes number {0} arrives .'.format(task_idx + 1))

        # Training loop
        model.train()
        for epoch in range(epochs):
            train_err = 0.0
            train_batches = 0
            start_time = time.time()

            for patterns, labels in tqdm(train_loader):
                targets = make_batch_one_hot(labels, 100).to(device)
                patterns = patterns.to(device)

                old_train = train_err
                train_err += train_fn(patterns, targets)

                if (train_batches % 100) == 1:
                    print(train_err - old_train)

                train_batches += 1

            # Validation after epoch
            acc_result, val_err, _, _ = get_accuracy(
                model,
                task_info.get_current_test_set(),
                device=device,
                required_top_k=[1, 5],
                return_detailed_outputs=False,
                criterion=BCELoss(),
                make_one_hot=True,
                n_classes=100,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8
            )
            print("Batch of classes {} out of {} batches".format(
                task_idx + 1, 100 // nb_cl))
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1,
                epochs,
                time.time() - start_time))
            print("  training loss:		{:.6f}".format(train_err / train_batches))
            print("  validation loss:		{:.6f}".format(val_err))
            print("  top 1 accuracy:		{:.2f} %".format(
                acc_result[0].item() * 100))
            print("  top 5 accuracy:		{:.2f} %".format(
                acc_result[1].item() * 100))
            scheduler.step()

        # Save model state each session
        torch.save(model.state_dict(),
                   'net_incr' + str(task_idx + 1) + '_of_' + str(100 // nb_cl))
        torch.save(model.feature_extractor.state_dict(),
                   'intermed_incr' + str(task_idx + 1) + '_of_' + str(100 // nb_cl))

        # Lines 220-242: Exemplars
        nb_protos_cl = int(ceil(nb_protos * 100. / nb_cl / (task_idx + 1)))

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
        class_means = torch.zeros((64, 100, 2), dtype=torch.float)
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
        print('Computing accuracy on the original batch of classes...')
        top1_acc_list_ori = icarl_accuracy_measure(task_info.get_task_test_set(0), class_means, val_fn,
                                                   top1_acc_list_ori, task_idx, 0, 'original',
                                                   make_one_hot=True, n_classes=100,
                                                   batch_size=batch_size, num_workers=8)

        top1_acc_list_cumul = icarl_accuracy_measure(task_info.get_cumulative_test_set(), class_means, val_fn,
                                                     top1_acc_list_cumul, task_idx, 0, 'cumul of',
                                                     make_one_hot=True, n_classes=100,
                                                     batch_size=batch_size, num_workers=8)
        map_whole = retrieval_performances(task_info.get_cumulative_test_set(), model, map_whole, task_idx)
    # Final save of the data
    torch.save(top1_acc_list_cumul, 'top1_acc_list_cumul_icarl_cl' + str(nb_cl))
    torch.save(top1_acc_list_ori, 'top1_acc_list_ori_icarl_cl' + str(nb_cl))
    torch.save(map_whole, 'map_list_cumul_icarl_cl' + str(nb_cl))
if __name__ == '__main__':
    main()
