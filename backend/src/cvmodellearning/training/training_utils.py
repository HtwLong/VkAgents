def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True) # clear grads efficiently
        outputs = model(images) # forward pass
        loss = criterion(outputs, targets) # compute batch loss
        loss.backward() # calculate gradients of loss function to know how much each weight and bias contributes to the loss
        optimizer.step() # adjust weights and biases based on calculated gradients

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc