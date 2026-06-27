import torch
from monai.metrics import DiceMetric
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy

def init_metrics(device):
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = BinaryJaccardIndex().to(device)
    acc_metric = BinaryAccuracy().to(device)
    return dice_metric, iou_metric, acc_metric

def train_epoch(model, train_loader, optimizer, criterion, device, writer, epoch):
    model.train()
    running_loss = 0.0
    dice_metric, iou_metric, acc_metric = init_metrics(device)
    all_probs = []
    all_targets = []
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device) # move to GPU/CPU
        masks = masks.to(device)
        optimizer.zero_grad() # reset gradients
        outputs = model(images) # forward pass
        loss = criterion(outputs, masks) # compute loss
        loss.backward() # backward pass
        optimizer.step() # update weights
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("loss/train", loss.item(), global_step)
        running_loss = running_loss + loss.item()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        dice_metric(y_pred=preds, y=masks)
        iou_metric(preds, masks)
        acc_metric(preds, masks)
        all_probs.append(probs.detach().cpu())
        all_targets.append(masks.detach().cpu())
        print(f"batch {batch_idx + 1}/{len(train_loader)} completed")
    train_loss = running_loss / len(train_loader)
    dice = dice_metric.aggregate().item()
    iou = iou_metric.compute().item()
    acc = acc_metric.compute().item()
    all_probs = torch.cat(all_probs).view(-1).numpy()
    all_targets = torch.cat(all_targets).view(-1).numpy()
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.0
    dice_metric.reset()
    iou_metric.reset()
    acc_metric.reset()
    return train_loss, dice, iou, acc, auc