from PIL import Image
import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from download_data import get_images_dir
from sklearn.model_selection import train_test_split

# expected img size
IMG_SIZE = 320

# ImageNet mean and std var
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class CustomDataset(Dataset):
    def __init__(self, bboxes_list):
        self.samples = bboxes_list
        self.transform = create_transform(IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img_tensor, bbox_tensor = preprocess_image_bbox(entry, self.transform)
        # Prepare the target dictionary
        target = {
            'boxes': bbox_tensor.unsqueeze(0),  # [N, 4]
            'labels': torch.ones((1,), dtype=torch.int64)  # cat = 1
        }
        return img_tensor, target

# resize images
# when converting to tensor, it automatically scales pixel values to floats in [0,1]
# standardize with pretrained training dataset mean and std var
def create_transform(img_size, mean, std):
    return T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
    ])

# adjust bounding boxes
# change pixel value datatype to float
def preprocess_image_bbox(entry, transform):
    images_dir = get_images_dir()
    img_path = os.path.join(images_dir, entry['filename'])
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    x, y, w, h = entry['bbox']
    # Convert to normalized coordinates
    bbox_norm = [
        x / orig_w, y / orig_h,
        (x + w) / orig_w, (y + h) / orig_h
    ]
    
    img_resized = transform(img)

    # Convert back to absolute coordinates for resized image (e.g. 320x320)
    img_target_w, img_target_h = img_resized.shape[2], img_resized.shape[1]  # tensor (C, H, W)
    bbox_abs = [
        bbox_norm[0] * img_target_w,
        bbox_norm[1] * img_target_h,
        bbox_norm[2] * img_target_w,
        bbox_norm[3] * img_target_h,
    ]
    
    return img_resized, torch.tensor(bbox_abs, dtype=torch.float32)


def splitBboxesTrainValTest(bboxes, train_size, val_size, test_size):
    if train_size+val_size+test_size != 1:
          raise Exception("Sum of Train, Val, Test percentage must be 1.") 
    total_val_test = val_size + test_size
    train_bboxes, temp_bboxes = train_test_split(bboxes, test_size=total_val_test, random_state=42)
    val_bboxes, test_bboxes = train_test_split(temp_bboxes, test_size=test_size/total_val_test, random_state=42)
    return train_bboxes, val_bboxes, test_bboxes

def getCocoCategories(classes, labels):
    category_ids = []
    for c in classes:
        for category in labels['categories']:
            if category['name'] == c:
                category_ids.append(category['id'])
                break
    return category_ids

# iterate through the labels.json to
# search for all images where its category is one we want (in category_ids)
# and store the image path, bounding box coordinates and category id as a dict in bboxes
def getBoundingBoxes(category_ids, id_to_filename_dict, labels):
    bboxes = []
    for ann in labels['annotations']:
        if ann['category_id'] in category_ids:
            img_filename = id_to_filename_dict[ann['image_id']]
            bbox = ann['bbox']  # COCO: [xmin, ymin, width, height]
            bboxes.append({
                'filename': img_filename,
                'bbox': bbox,
                'category_id': ann['category_id']})
    return bboxes