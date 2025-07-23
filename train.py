import torch
import logging
from torchvision.ops.boxes import box_iou

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
        if i % print_freq == 0:
            logging.info(f"Epoch {epoch}, step {i}, loss: {losses.item():.4f}")
    return running_loss / len(data_loader)

def evaluate_model(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                pred_labels = output['labels'].cpu()
                keep = pred_scores > score_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                all_preds.append({'boxes': pred_boxes, 'labels': pred_labels})
                all_targets.append({'boxes': gt_boxes, 'labels': gt_labels})
    # Simplified single-class mAP@0.5 calculation
    tp, fp, fn = 0, 0, 0
    for pred, target in zip(all_preds, all_targets):
        pred_boxes = pred['boxes']
        gt_boxes = target['boxes']
        if len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue
        ious = box_iou(pred_boxes, gt_boxes) if len(pred_boxes) and len(gt_boxes) else torch.zeros((0,0))
        matched_gt = set()
        for i, pb in enumerate(pred_boxes):
            max_iou, max_idx = (ious[i].max(0) if ious.shape[1] > 0 else (torch.tensor(0.), torch.tensor(-1)))
            if max_iou > iou_threshold and max_idx.item() not in matched_gt:
                tp += 1
                matched_gt.add(max_idx.item())
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    ap = precision * recall  # Simplified AP
    logging.info(f"Eval: TP={tp}, FP={fp}, FN={fn}, Precision={precision:.4f}, Recall={recall:.4f}, AP@0.5={ap:.4f}")
    return ap
