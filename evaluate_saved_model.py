import torch
from torch.utils.data import DataLoader
from preprocessing import BUSIDataset, get_patient_pairs, split_patient_records, img_transform, mask_transform
from fold_utils import records_to_paths
from test_evaluation import evaluate_model_on_test
from visualization import show_prediction_overlay
from model import CResUNet
from UNet import UNet
import shutil
import os

model = CResUNet()
model_path = "CResUNet_model.pth"
root_dir = "BUSI_Dataset"
test_ratio = 0.2
seed = 42
batch_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def save_test_dataset(test_records, output_dir):
    for record in test_records:
        class_name = record["class_name"]
        patient_id = record["patient_id"]
        image_path = record["image_path"]
        mask_paths = record["mask_paths"]
        # create the same folder structure as the original dataset
        patient_output_dir = os.path.join(output_dir, class_name, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        shutil.copy2(image_path, os.path.join(patient_output_dir, os.path.basename(image_path))) # copy original image
        for mask_path in mask_paths: # copy original masks
            shutil.copy2(mask_path, os.path.join(patient_output_dir, os.path.basename(mask_path)))

def results_for_all_test_set(model, model_path):
    test_image_paths, test_mask_paths = records_to_paths(test_records)
    print(f"test samples: {len(test_image_paths)}")
    test_dataset = BUSIDataset(test_image_paths, test_mask_paths, image_transform=img_transform(),
                               mask_transform=mask_transform(), augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loss, test_dice, test_iou, test_acc, test_auc, results = evaluate_model_on_test(model=model, model_path=model_path,
                                                                                         test_loader=test_loader, device=device)
    return test_dice, test_iou, results

def results_per_class(model, model_path):
    # evaluate model separately for each class
    test_root_dir = "BUSI_data_test"
    all_test_records = get_patient_pairs(test_root_dir)
    classes = ["normal", "benign", "malignant"]
    for class_name in classes:
        print("\n")
        print(f"testing on: {class_name}")
        # keep only records from the current class
        class_records = [record for record in all_test_records if record["class_name"] == class_name]
        test_image_paths, test_mask_paths = records_to_paths(class_records)
        print(f"test samples: {len(test_image_paths)}")
        test_dataset = BUSIDataset(test_image_paths, test_mask_paths, image_transform=img_transform(),
                                   mask_transform=mask_transform(), augmentation=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        # evaluate trained model
        test_loss, test_dice, test_iou, test_acc, test_auc, results = evaluate_model_on_test(model=model, model_path=model_path,
                                                                                             test_loader=test_loader, device=device)
        print(f"dice: {test_dice:.4f}")
        print(f"IoU: {test_iou:.4f}")
        print(f"accuracy: {test_acc:.4f}")
        print(f"AUC: {test_auc:.4f}")

        results_sorted = sorted(results, key=lambda x: x["dice"], reverse=True)
        show_prediction_overlay(results_sorted[:]) # best dice metrics cases



all_records = get_patient_pairs(root_dir)
train_records, test_records = split_patient_records(all_records, test_ratio=test_ratio, seed=seed)
save_test_dataset(test_records, output_dir="BUSI_data_test")
test_dice, test_iou, results = results_for_all_test_set(model, model_path)
print("results for all test dataset:")
print(f"dice: {test_dice:.4f}")
print(f"IoU: {test_iou:.4f}")
print("results per class:")
results_per_class(model, model_path)
#results_sorted = sorted(results, key=lambda x: x["dice"], reverse=True)
#show_prediction_overlay(results_sorted[:]) # best dice metrics cases
#show_prediction_overlay(results_sorted[-5:]) # worst dice metrics cases



