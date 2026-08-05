import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np

@torch.inference_mode()
def evaluate(classes, model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_top5_correct = 0
    total = 0

    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        if outputs.shape[1] >= 5:
            top5 = outputs.topk(5, dim=1).indices
            running_top5_correct += top5.eq(targets[:, None]).any(dim=1).sum().item()
        total += images.size(0)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_correct / total if total > 0 else 0.0

    y_true = np.concatenate(all_targets) if all_targets else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])

    # Detailed metrics
    metrics = {}
    if y_true.size > 0:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(classes))), zero_division=0, average=None
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(classes))), zero_division=0, average="macro"
        )
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(classes))), zero_division=0, average="micro"
        )

        report_dict = classification_report(
            y_true, y_pred, labels=list(range(len(classes))), target_names=classes, 
            zero_division=0, digits=4, output_dict=True
        )

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

        metrics.update({
            "accuracy": float(epoch_acc),
            "loss": float(epoch_loss),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "support_per_class": support.tolist(),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "classification_report_dict": report_dict,
            "confusion_matrix": cm.tolist(),
        })
        if len(classes) >= 5:
            metrics["top5_acc"] = float(running_top5_correct / total)
    else:
        metrics.update({"accuracy": 0.0, "loss": epoch_loss})

    return epoch_loss, epoch_acc, metrics
