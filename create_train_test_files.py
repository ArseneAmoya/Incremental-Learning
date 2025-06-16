from glob import glob

train_images = glob("data/CUB_200_2011/train/*/*")
print(f"Found {len(train_images)} training images")
train_images = [image.replace("data/CUB_200_2011/", "").replace('\\', '/')  for image in train_images]
test_images = glob("data/CUB_200_2011/test/*/*")
print(f"Found {len(test_images)} testing images")
image_dict = {}
print(f"Total images in train and test: {len(train_images) + len(test_images)}")
test_images = [image.replace("data/CUB_200_2011/", "").replace('\\', '/') for image in test_images]
print(f"example train image: {train_images[0] if train_images else 'None'}")
with open("data/CUB_200_2011/train.txt", "w") as train_file, open("data/CUB_200_2011/test.txt", "w") as test_file, open("data/CUB_200_2011/image_class_labels.txt", "r") as label_file, open("data/CUB_200_2011/images.txt", "r") as images_file:
    
    for line in images_file:
        line = line.strip()
        if not line:
            continue
        id, image = line.split(" ")
        image_dict[id] = image
        
    for line in label_file:
        line = line.strip()
        if not line:
            continue
        image, label = line.split(" ")
        if f"train/{image_dict[image]}" in train_images:
            train_file.write(f"train/{image_dict[image]} {label}\n")
        elif f"test/{image_dict[image]}" in test_images:
            test_file.write(f"test/{image_dict[image]} {label}\n")
        else:
            print(f"Image {image_dict[image]} not found in train or test images")
            break
    
