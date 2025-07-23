import os
import fiftyone.zoo as foz
import fiftyone as fo
import shutil

def download_data(classes, split, label_types, max_samples):

    # set the download path to be in the current directory under a new fiftyone folder
    image_download_path = os.path.join(os.getcwd(), "fiftyone")
    fo.config.dataset_zoo_dir = image_download_path

    # clear old data before downloading new data
    if os.path.isdir(image_download_path): 
        shutil.rmtree(image_download_path)

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

def get_dataset_dir():
    return os.path.join(os.getcwd(), "fiftyone", "coco-2017", "train")

def get_images_dir():
     return os.path.join(os.getcwd(), "fiftyone", "coco-2017", "train", "data")

def get_labels_json_path():
    return os.path.join(os.getcwd(), "fiftyone", "coco-2017", "train", "labels.json")
