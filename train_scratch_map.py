# File: train_scratch.py
import time
from functools import partial
from typing import List

import torch
from torch import Tensor
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets.cifar import CIFAR100
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchmetrics.functional.retrieval import retrieval_average_precision

from cl_dataset_tools import NCProtocol
from cl_strategies import icarl_cifar100_augment_data
from cl_metrics_tools import get_accuracy
from models import make_icarl_net
from models.icarl_net import initialize_icarl_net
from utils import (
    get_dataset_per_pixel_mean,
    make_theano_training_function,
    make_theano_feature_extraction_function,
    make_batch_one_hot
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device {device}")

# Data transforms and mean subtraction
base_transform = transforms.Compose([transforms.ToTensor()])
per_pixel_mean = get_dataset_per_pixel_mean(
    CIFAR100('./data/cifar100', train=True, download=True, transform=base_transform)
)
def transform1(x: Tensor) -> Tensor:
    return x - per_pixel_mean

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transform1,
    icarl_cifar100_augment_data,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transform1,
])

# Main training-from-scratch function

def train_scratch():
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
    map_list            = torch.zeros(100 // nb_cl, epochs)

    # Prepare continual-learning protocol
    protocol = NCProtocol(
        CIFAR100('./data/cifar100', train=True, download=True, transform=transform_train),
        CIFAR100('./data/cifar100', train=False, download=True, transform=transform_test),
        n_tasks=100 // nb_cl,
        shuffle=True,
        fixed_class_order=fixed_class_order
    )

    # Model initialization
    model = make_icarl_net(num_classes=100)
    model.apply(initialize_icarl_net)
    model = model.to(device)
    criterion = BCELoss()
    sh_lr = lr_old

    # Keep track of seen training data
    cumulative_datasets: List[Dataset] = []

    # Iterate over sessions
    for task_idx, (train_ds, task_info) in enumerate(protocol):
        print(f"Classes in this batch: {task_info.classes_in_this_task}")

        # Save cumulative metrics so far
        torch.save(top1_acc_list_cumul, f'top1_acc_list_cumul_cl{nb_cl}.pt')
        torch.save(top1_acc_list_ori,   f'top1_acc_list_ori_cl{nb_cl}.pt')
        torch.save(map_list,            f'map_list_scratch_cl{nb_cl}.pt')

        # Build cumulative training set
        cumulative_datasets.append(train_ds)
        train_dataset = ConcatDataset(cumulative_datasets)
        train_loader  = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        # Optimizer, scheduler, and training function
        optimizer = torch.optim.SGD(
            model.parameters(), lr=sh_lr,
            weight_decay=wght_decay, momentum=0.9
        )
        scheduler = MultiStepLR(optimizer, lr_strat, gamma=1.0 / lr_factor)
        train_fn = partial(
            make_theano_training_function,
            model, criterion, optimizer, device=device
        )
        feature_fn = partial(
            make_theano_feature_extraction_function,
            
            model, 'feature_extractor', device=device
        )

        # Pre-training validation
        print("\nBefore first epoch:")
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
        print(f"  validation loss:		{val_err:.6f}")
        print(f"  top 1 accuracy:		{acc_result[0].item()*100:.2f} %")
        print(f"  top 5 accuracy:		{acc_result[1].item()*100:.2f} %")

        print(f"Batch of classes number {task_idx+1} arrives.")

        # Training loop
        model.train()
        for epoch in range(epochs):
            train_err, train_batches = 0.0, 0
            start_time = time.time()

            for patterns, labels in train_loader:
                patterns = patterns.to(device)
                targets = make_batch_one_hot(labels, 100).to(device)
                train_err += train_fn(patterns, targets)
                train_batches += 1

            # Validation
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
            print(f"Batch {task_idx+1}/{100//nb_cl}, Epoch {epoch+1}/{epochs}, Time {time.time()-start_time:.3f}s")
            print(f"  training loss:		{train_err/train_batches:.6f}")
            print(f"  validation loss:		{val_err:.6f}")
            print(f"  top 1 accuracy:		{acc_result[0].item()*100:.2f} %")
            print(f"  top 5 accuracy:		{acc_result[1].item()*100:.2f} %")

            # Compute retrieval mAP
            model.eval()
            all_feats, all_labels = [], []
            with torch.no_grad():
                test_loader = DataLoader(
                    task_info.get_current_test_set(),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8
                )
                for patterns, labels in test_loader:
                    feats = feature_fn(patterns.to(device))
                    all_feats.append(feats.cpu())
                    all_labels.append(labels)
            feats = torch.cat(all_feats)
            labels = torch.cat(all_labels).int()
            map_score = retrieval_average_precision(feats, labels)
            print(f"  retrieval mAP:		{map_score*100:.2f} %")
            map_list[task_idx, epoch] = map_score
            torch.save(map_list, f'map_list_scratch_cl{nb_cl}.pt')

            scheduler.step()

        # Save model state and metrics
        torch.save(model.state_dict(), f'net_scratch_session{task_idx+1}.pt')
        torch.save(model.feature_extractor.state_dict(), f'intermed_scratch_session{task_idx+1}.pt')
        torch.save(top1_acc_list_cumul, f'top1_cumul_cl{nb_cl}.pt')
        torch.save(top1_acc_list_ori,   f'top1_ori_cl{nb_cl}.pt')

    # Final save of metrics
    torch.save(map_list,            f'map_list_scratch_cl{nb_cl}.pt')
    torch.save(top1_acc_list_cumul, f'top1_cumul_cl{nb_cl}.pt')
    torch.save(top1_acc_list_ori,   f'top1_ori_cl{nb_cl}.pt')

if __name__ == '__main__':
    train_scratch()
