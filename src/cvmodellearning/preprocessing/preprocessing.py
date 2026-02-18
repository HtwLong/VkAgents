import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


# Dataset class for loading images and labels from CSV
class CocoImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, class_to_idx=None):
        self.data = pd.read_csv(csv_file, header=0, names=['image_filename', 'labels'])
        self.root_dir = root_dir
        self.transform = transform

        if class_to_idx is not None:
            self.class_to_idx = dict(class_to_idx)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            # Map labels to ints using the shared mapping; verify no missing
            self.data['label_enc'] = self.data['labels'].map(self.class_to_idx)
            if self.data['label_enc'].isnull().any():
                missing = self.data[self.data['label_enc'].isnull()]['labels'].unique().tolist()
                raise ValueError(f"Found unknown labels {missing} not in class_to_idx {self.class_to_idx} but in {self.data['label_enc']}")
        else:
            # Fallback: fit a LabelEncoder locally (not recommended across splits)
            self.le = LabelEncoder()
            self.data['label_enc'] = self.le.fit_transform(self.data['labels'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx]['image_filename'])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx]['label_enc'])
        if self.transform:
            image = self.transform(image)
        return image, label
