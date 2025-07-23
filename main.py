import logging
import torch
from torch.utils.data import DataLoader

from download_data import download_data, get_labels_json_path
from preprocess import CustomDataset, getCocoCategories, getBoundingBoxes, splitBboxesTrainValTest
from model_utils import get_ssdlite_mobilenet_v3
from train import train_one_epoch, evaluate_model
from hyperparams import hyperparams

# Set up logging
logging.basicConfig(level=logging.INFO)

# Apple M1/M2/M3/M4 device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logging.info("Using MPS (Apple Silicon GPU) device")
else:
    device = torch.device("cpu")
    logging.info("Using CPU device")

# ------------------------------
# Variables & Setup
# ------------------------------
split = "train"
max_samples = 1000
label_types = ["detections"]
classes = ["cat"]
batch_size = 4

# Download data using fiftyone
download_data(classes, split, label_types, max_samples)

# Preprocess & Dataset Setup
import json
with open(get_labels_json_path(), "r") as f:
    labels = json.load(f)
category_ids = getCocoCategories(classes, labels)
id_to_filename_dict = {img['id']: img['file_name'] for img in labels['images']}
bboxes = getBoundingBoxes(category_ids, id_to_filename_dict, labels)
train_bboxes, val_bboxes, test_bboxes = splitBboxesTrainValTest(bboxes, 0.7, 0.15, 0.15)
train_dataset = CustomDataset(train_bboxes)
val_dataset = CustomDataset(val_bboxes)
test_dataset = CustomDataset(test_bboxes)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)), drop_last=True)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)), drop_last=True)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)), drop_last=True)


# Model setup
num_classes = len(classes) + 1
model = get_ssdlite_mobilenet_v3(num_classes=num_classes)
model.to(device)

# Hyperparameter Search and Training Loop
best_map = 0
best_params = None
for params in hyperparams:
    lr = params['lr']
    epochs = params['epochs']
    opt_name = params['optimizer']
    weight_decay = params['weight_decay']

    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {opt_name}")

    logging.info(f"--- Training: optimizer={opt_name}, lr={lr}, weight_decay={weight_decay}, epochs={epochs} ---")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        logging.info(f"Epoch {epoch} finished. Average train loss: {train_loss:.4f}")

    val_map = evaluate_model(model, val_loader, device)
    logging.info(f"Validation mAP@0.5: {val_map:.4f} for params {params}")

    if val_map > best_map and val_map > 0.5:
        best_map = val_map
        best_params = params
        torch.save(model.state_dict(), "best_ssdlite_model.pth")
        logging.info(f"New best model saved with mAP {best_map:.4f}")
        break
    else:
        logging.info(f"Validation mAP {val_map:.4f} did not improve best mAP {best_map:.4f}")

# Final evaluation on test data
if best_params:
    model.load_state_dict(torch.load("best_ssdlite_model.pth"))
    logging.info(f"Best model weights loaded with params {best_params}")
test_map = evaluate_model(model, test_loader, device)
logging.info(f"Final Test mAP@0.5: {test_map:.4f}") 
