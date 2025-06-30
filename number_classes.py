from DatasetModule import MyData
from torchvision import transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])
train_dataset = MyData('./data/CUB_200_2011', label_txt='data/CUB_200_2011/train.txt', transform=transform)

from collections import Counter

# Récupère tous les labels du dataset
all_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
label_counts = Counter(all_labels)
counts = {}
# Affiche les classes qui n'ont pas exactement 30 éléments
for label, count in sorted(label_counts.items()):
    counts[label] = count

print(counts)    
