import fiftyone.zoo as foz
import fiftyone as fo
import os
import shutil



### ------------------------------
### Set env variables
### ------------------------------
image_download_path = os.path.join(os.getcwd(), "fiftyone")
fo.config.dataset_zoo_dir = image_download_path
split = "train"
max_samples = 10
label_types = ["detections"]
classes = ["cat"]

# clear old data before downloading new data
if os.path.isdir(image_download_path): 
    shutil.rmtree(image_download_path)


### ------------------------------
### Download Dataset 
### ------------------------------

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split=split,
    label_types=label_types,
    classes=classes,
    max_samples=max_samples,
    shuffle=True,
    dataset_name="custom-cat-dataset",
)

# show how much storage the data takes
dataset.compute_metadata(overwrite=True)
stats = dataset.stats(include_media=True)
fo.pprint(stats)

### ------------------------------
### Data Preprocessing
### ------------------------------

# resize images (necessary)
# change pixel value datatype to float
# scale pixel values to [0,1] (necessary)
# standardize with pretrained training dataset mean and std var
# 





### ------------------------------
### 
### ------------------------------




