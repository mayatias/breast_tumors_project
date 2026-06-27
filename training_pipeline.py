import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.losses import DiceLoss
import matplotlib
from tqdm.auto import tqdm
from evaluation import evaluate
from fold_utils import make_stratified_patient_folds, records_to_paths
from model import CResUNet
from preprocessing import BUSIDataset, get_patient_pairs, img_transform, mask_transform, split_patient_records
from train import train_epoch
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_loss_plot(output_path, train_losses, val_losses=None, title="Loss vs Epoch"):
    epochs = range(1, len(train_losses) + 1)
    fig = plt.figure()
    fig.set_size_inches(8, 5)
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    if val_losses is not None:
        plt.plot(epochs, val_losses, marker="o", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def average_histories(histories):
    if not histories:
        return []
    max_len = max(len(history) for history in histories)
    averaged = []
    for epoch_idx in range(max_len):
        epoch_values = [history[epoch_idx] for history in histories if epoch_idx < len(history)]
        averaged.append(sum(epoch_values) / len(epoch_values))
    return averaged

def run_fold(fold, fold_records, n_splits, batch_size, epochs, lr, device):
    if n_splits == 1:
        train_fold_records, val_records = fold_records[0]
    else:
        val_records = fold_records[fold]
        train_fold_records = []
        for other_fold in range(n_splits):
            if other_fold != fold:
                train_fold_records.extend(fold_records[other_fold])
    fold_train_image_paths, fold_train_mask_paths = records_to_paths(train_fold_records)
    fold_val_image_paths, fold_val_mask_paths = records_to_paths(val_records)
    train_dataset = BUSIDataset(fold_train_image_paths, fold_train_mask_paths, image_transform=img_transform(),
                                mask_transform=mask_transform(), augmentation=True)
    val_dataset = BUSIDataset(fold_val_image_paths, fold_val_mask_paths, image_transform=img_transform(),
                              mask_transform=mask_transform(), augmentation=False)
    print(f"fold {fold + 1} training dataset size (after augmentation): {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = CResUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = DiceLoss(sigmoid=True)
    best_dice = 0
    best_iou = 0
    best_acc = 0
    best_auc = 0
    train_losses = []
    val_losses = []
    epoch_bar = tqdm(range(epochs), desc=f"fold {fold + 1}/{n_splits}", leave=False)
    writer = SummaryWriter(log_dir=f"runs/fold_{fold + 1}")
    for epoch in epoch_bar:
        train_loss, train_dice, train_iou, train_acc, train_auc = train_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        train_losses.append(train_loss)
        print(f"train epoch {epoch+1}/{epochs} | " f"train loss: {train_loss:.4f} | train dice: {train_dice:.4f} | " 
              f"train IoU: {train_iou:.4f} | train accuracy: {train_acc:.4f} | train AUC: {train_auc:.4f}")
        val_loss, val_dice, val_iou, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        writer.add_scalar("loss/validation", val_loss, epoch)
        writer.add_scalar("metrics/dice_validation", val_dice, epoch)
        writer.add_scalar("metrics/iou_validation", val_iou, epoch)
        writer.add_scalar("metrics/accuracy_validation", val_acc, epoch)
        writer.add_scalar("metrics/auc_validation", val_auc, epoch)
        val_losses.append(val_loss)
        print(f"val epoch {epoch+1}/{epochs} | " f"val loss: {val_loss:.4f} | val dice: {val_dice:.4f} | " 
              f"val IoU: {val_iou:.4f} | val accuracy: {val_acc:.4f} | val AUC: {val_auc:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            best_acc = val_acc
            best_auc = val_auc
            torch.save(model.state_dict(), f"best_fold_{fold+1}.pth")
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_dice=f"{val_dice:.4f}", best_dice=f"{best_dice:.4f}")
    plot_path = f"loss_curve_fold_{fold + 1}.png"
    save_loss_plot(plot_path, train_losses, val_losses, title=f"Fold {fold + 1} loss vs epoch")
    writer.close()
    return best_dice, best_iou, best_acc, best_auc, train_losses, val_losses

def train_final_model(train_records, batch_size, epochs, lr, device):
    train_image_paths, train_mask_paths = records_to_paths(train_records)
    train_dataset = BUSIDataset(train_image_paths, train_mask_paths, image_transform=img_transform(),
                                mask_transform=mask_transform(), augmentation=True)
    print(f"final training dataset size (after augmentation): {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = CResUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = DiceLoss(sigmoid=True)
    train_losses = []
    epoch_bar = tqdm(range(epochs), desc="final train", leave=False)
    writer = SummaryWriter(log_dir="runs/final_model")
    for epoch in epoch_bar:
        train_loss, train_dice, train_iou, train_acc, train_auc = train_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        train_losses.append(train_loss)
        print(f"final train epoch {epoch+1}/{epochs} | " f"train loss: {train_loss:.4f} | train dice: {train_dice:.4f} | "
              f"train IoU: {train_iou:.4f} | train accuracy: {train_acc:.4f} | train AUC: {train_auc:.4f}")
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", train_dice=f"{train_dice:.4f}")
    torch.save(model.state_dict(), "final_model.pth")
    save_loss_plot("final_train_loss_curve.png", train_losses, title="Final training loss vs epoch")
    writer.close()
    return "final_model.pth", train_losses

def evaluate_model_on_test(model_path, test_loader, device):
    model = CResUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = DiceLoss(sigmoid=True)
    test_loss, test_dice, test_iou, test_acc, test_auc = evaluate(model, test_loader, criterion, device)
    print("\n test-set results:")
    print(f"test loss: {test_loss:.4f}")
    print(f"test dice: {test_dice:.4f}")
    print(f"test IoU: {test_iou:.4f}")
    print(f"test accuracy: {test_acc:.4f}")
    print(f"test AUC: {test_auc:.4f}")
    return test_loss, test_dice, test_iou, test_acc, test_auc

def run_experiment(root_dir, n_splits, batch_size, epochs, lr, test_ratio, seed, device):
    all_records = get_patient_pairs(root_dir)
    train_records, test_records = split_patient_records(all_records, test_ratio=test_ratio, seed=seed)
    if n_splits == 1:
        train_records, val_records = split_patient_records(train_records, test_ratio=0.2, seed=seed)
        fold_records = [(train_records, val_records)]
    else:
        fold_records = make_stratified_patient_folds(train_records, n_splits=n_splits, seed=seed)
    train_image_paths, _ = records_to_paths(train_records)
    test_image_paths, test_mask_paths = records_to_paths(test_records)
    print(f"train samples: {len(train_image_paths)}")
    print(f"test samples:  {len(test_image_paths)}")
    test_dataset = BUSIDataset(test_image_paths, test_mask_paths, image_transform=img_transform(),
                               mask_transform=mask_transform(), augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dice_scores = []
    iou_scores = []
    acc_scores = []
    auc_scores = []
    fold_train_histories = []
    fold_val_histories = []
    fold_bar = tqdm(range(n_splits), desc="cross-val folds", leave=True)
    for fold in fold_bar:
        print(f"\nfold {fold + 1}\n")
        best_dice, best_iou, best_acc, best_auc, train_losses, val_losses = run_fold(
            fold=fold,
            fold_records=fold_records,
            n_splits=n_splits,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            device=device,
        )
        dice_scores.append(best_dice)
        iou_scores.append(best_iou)
        acc_scores.append(best_acc)
        auc_scores.append(best_auc)
        fold_train_histories.append(train_losses)
        fold_val_histories.append(val_losses)
        print(f"\nbest dice for fold {fold+1}: {best_dice:.4f}")
        fold_bar.set_postfix(best_dice=f"{best_dice:.4f}")
    print("\nfinal results:")
    print("dice: ", sum(dice_scores) / len(dice_scores))
    print("IoU:  ", sum(iou_scores) / len(iou_scores))
    print("accuracy:  ", sum(acc_scores) / len(acc_scores))
    print("AUC:  ", sum(auc_scores) / len(auc_scores))

    avg_train_losses = average_histories(fold_train_histories)
    avg_val_losses = average_histories(fold_val_histories)
    if avg_train_losses and avg_val_losses:
        save_loss_plot("cross_validation_loss_curve.png", avg_train_losses, avg_val_losses,
                       title="Average cross-validation loss vs epoch")

    final_model_path, _ = train_final_model(train_records=train_records, batch_size=batch_size,
                                           epochs=epochs, lr=lr, device=device)
    evaluate_model_on_test(final_model_path, test_loader, device)