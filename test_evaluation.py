import torch
from monai.metrics import DiceMetric
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy
from tqdm.auto import tqdm
from monai.losses import DiceLoss

def init_metrics(device):
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = BinaryJaccardIndex().to(device)
    acc_metric = BinaryAccuracy().to(device)
    return dice_metric, iou_metric, acc_metric

def test_evaluate(model, val_loader, criterion, device):
    model.eval()
    dice_metric, iou_metric, acc_metric = init_metrics(device)
    running_loss = 0.0
    all_probs = []
    all_targets = []
    results = []
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="eval batches", leave=False)):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            dice_metric(y_pred=preds, y=masks)
            iou_metric(preds, masks)
            acc_metric(preds, masks)
            all_probs.append(probs.cpu())
            all_targets.append(masks.cpu())
            # calculate dice score for each image
            for i in range(images.shape[0]):
                pred = preds[i]
                target = masks[i]
                intersection = (pred * target).sum()
                dice_img = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
                results.append({"dice": dice_img.item(), "image": images[i].cpu(), "mask": target.cpu(), "pred": pred.cpu()})
            print(f"eval batch {batch_idx + 1}/{len(val_loader)} completed")
    val_loss = running_loss / len(val_loader)
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
    return val_loss, dice, iou, acc, auc, results

def evaluate_model_on_test(model, model_path, test_loader, device):
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = DiceLoss(sigmoid=True)
    test_loss, test_dice, test_iou, test_acc, test_auc, results = test_evaluate(model, test_loader, criterion, device)
    print("\n test-set results:")
    print(f"test dice: {test_dice:.4f}")
    print(f"test IoU: {test_iou:.4f}")
    print(f"test accuracy: {test_acc:.4f}")
    print(f"test AUC: {test_auc:.4f}")
    return test_loss, test_dice, test_iou, test_acc, test_auc, results