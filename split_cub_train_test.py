import os
import shutil
from glob import glob
path_split_dict = {}
images_path_dict = {}
with open("data/CUB_200_2011/train_test_split.txt","r") as split_file, open("data/CUB_200_2011/images.txt", "r") as output_file:
    for line in split_file:
        line = line.strip().split()
        if len(line) == 2:
            image_id, split = line
            path_split_dict[image_id] = split
    line= split_file.readline().strip().split()
    print(len(path_split_dict), "images in split file")
    for line in output_file:
        line = line.strip().split()
        if len(line) == 2:
            image_id, image_path = line
            images_path_dict[image_id] = image_path
    print(len(images_path_dict), "images in images file")

    for image_id, split in path_split_dict.items():
        if split == "1":
            split_dir = "train"
        else:
            split_dir = "test"
        
        image_path = images_path_dict.get(image_id, None)
        if image_path is not None:
            source_path = os.path.join("data/CUB_200_2011/images", image_path)
            target_dir = os.path.join("data/CUB_200_2011", split_dir)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.copy(source_path, target_dir)
            print(f"Copied {source_path} to {target_dir}")
    
    all_resulting_images = glob("data/CUB_200_2011/train/*") + glob("data/CUB_200_2011/test/*")
    print(f"Total images in train: {len(glob('data/CUB_200_2011/train/*'))}")
    print(f"Total images in test: {len(glob('data/CUB_200_2011/test/*'))}")
    print(f"Total images processed: {len(all_resulting_images)}")

    
