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
from models.icarl_net import initialize_icarl_net
from models.icarl_net import make_icarl_net


from utils import retrieval_performances

from cl_metrics_tools import get_accuracy

parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
args = parser.parse_args()

tasks = 10
n_epochs = args.epochs
top_k_accuracies = [1, 5]
lr_old     = 2.             # Initial learning rate
lr_strat   = [49, 63]       # Epochs where learning rate gets decreased
lr_factor  = 5.             # Learning rate decrease factor
wght_decay = 0.00001 
torch.manual_seed(2000)

sh_lr = lr_old

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


fixed_class_order = [87,  0, 52, 58, 44, 91, 68, 97, 51, 15,
                        94, 92, 10, 72, 49, 78, 61, 14,  8, 86,
                        84, 96, 18, 24, 32, 45, 88, 11,  4, 67,
                        69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
                        17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
                        1, 28,  6, 46, 62, 82, 53,  9, 31, 75,
                        38, 63, 33, 74, 27, 22, 36,  3, 16, 21,
                        60, 19, 70, 90, 89, 43,  5, 42, 65, 76,
                        40, 30, 23, 85,  2, 95, 56, 48, 71, 64,
                        98, 13, 99,  7, 34, 55, 54, 26, 35, 39]

metrics = torch.zeros(tasks, 10)
losses = torch.zeros(tasks, n_epochs, 2)
time_list = torch.zeros(tasks, n_epochs)

def main():
    protocol = NCProtocol(CIFAR100('./data/cifar100', train=True, download=True, transform=transform),
                          CIFAR100('./data/cifar100', train=False, download=True, transform=transform_test),
                          n_tasks=tasks,fixed_class_order=fixed_class_order, seed=None, shuffle=True)

    model = make_icarl_net(num_classes=100)#ResNet18Cut().to(device)# resnet18(pretrained=False, num_classes=100).to(device)
    model.apply(initialize_icarl_net)
    model = model.to(device)
    train_dataset: Dataset
    task_info: NCProtocolIterator
    map_list = torch.zeros(tasks, 2)
    for task_idx, (train_dataset, task_info) in enumerate(protocol):
        print()
        print('-------------------------------------------------------------------------------')
        print('Task', task_idx, 'started')
        
        print('Classes in this batch:', task_info.classes_in_this_task)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

        optimizer = torch.optim.SGD(model.parameters(), lr=sh_lr, weight_decay=wght_decay, momentum=0.9)
        criterion = BCELoss()
        scheduler = MultiStepLR(optimizer, lr_strat, gamma=1.0/lr_factor)



        for epoch in range(n_epochs):
            start_time = time.time()
            epoch_loss = 0
                  # Sets model in train mode
            model.train()
            for patterns, labels in train_loader:

                targets = make_batch_one_hot(labels, 100)

                # Clear grad
                optimizer.zero_grad()

                # Send data to device
                patterns = patterns.to(device)
                targets = targets.to(device)

                # Forward
                output = model(patterns)

                # Loss
                loss = criterion(output, targets)
                epoch_loss += loss.item()

                # Backward
                loss.backward()

                # Update step
                optimizer.step()
            
            scheduler.step()
            epoch_time = time.time() - start_time
            time_list[task_idx, epoch] = epoch_time
            losses[task_idx, epoch, 0] = epoch_loss / len(train_loader)
            print('Epoch', epoch, 'completed in', epoch_time, 'seconds')
            # ---- Validation ----
            model.eval()
            with torch.no_grad():
                val_accuracies, val_loss, _, _ = get_accuracy(model, task_info.get_cumulative_test_set(), device=device,
                                                 required_top_k=[1, 5], return_detailed_outputs=False,
                                                 criterion=BCELoss(), make_one_hot=True, n_classes=100,
                                                 batch_size=128, shuffle=False, num_workers=8)
                losses[task_idx, epoch, 1] = val_loss
            print(f'Epoch {epoch} train loss: {epoch_loss/len(train_loader):.5f} validation loss {val_loss:.5f} training time {epoch_time:.3f} seconds')    
        print('Task', task_idx, 'ended')

        top_train_accuracies, _, _, _ = get_accuracy(model,
                                                  task_info.swap_transformations().get_cumulative_training_set(),
                                                  device=device, required_top_k=top_k_accuracies, batch_size=128)

        top_train_accuracies_current, _, _, _ = get_accuracy(model,
                                                  task_info.swap_transformations().get_current_training_set(),
                                                  device=device, required_top_k=top_k_accuracies, batch_size=128)
        top_test_accuracies, _, _, _ = get_accuracy(model, task_info.get_cumulative_test_set(), device=device,
                                                 required_top_k=top_k_accuracies, batch_size=128)
        
        top_test_accuracies_current, _, _, _ = get_accuracy(model, task_info.get_current_test_set(), device=device,
                                                    required_top_k=top_k_accuracies, batch_size=128)
        map_list = retrieval_performances(task_info.get_cumulative_test_set(), model, map_list, task_idx, batch_size=128, current_classes=task_info.classes_in_this_task)

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

        torch.save(model.state_dict(), 'net_naive_1'+str(task_idx+1)+'_of_'+str(task_idx))
        torch.save(model.feature_extractor.state_dict(), 'intermed_naive_1'+str(task_idx+1)+'_of_'+str(task_idx))


    metrics_df = pd.DataFrame(metrics.numpy(), columns= ['top1_train_current', 'top1_train_cumul', 'top1_test_current', 'top1_test_cumul',
                                                                    'top5_train_current', 'top5_train_cumul', 'top5_test_current', 'top5_test_cumul',
                                                                    'map_cumul', 'map_current'])
    metrics_df.to_csv('metrics.csv', index=False)

    time_df = pd.DataFrame(time_list.numpy(), columns=None)
    losses_df = pd.DataFrame(losses.view(tasks*n_epochs, 2).numpy(), columns=['loss', 'val_loss'])
    #concat_df = pd.concat([time_df, losses_df], axis=1)
    losses_df.to_csv('losses.csv', index=False)
    time_df.to_csv('time.csv', index=False)



if __name__ == '__main__':
    main()
