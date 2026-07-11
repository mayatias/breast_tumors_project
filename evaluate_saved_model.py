import torch
from torch.utils.data import DataLoader
from preprocessing import BUSIDataset, get_patient_pairs, split_patient_records, img_transform, mask_transform
from fold_utils import records_to_paths
from test_evaluation import evaluate_model_on_test
from visualization import show_prediction_overlay

root_dir = "BUSI_Dataset"
test_ratio = 0.2
seed = 42
batch_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

all_records = get_patient_pairs(root_dir)
train_records, test_records = split_patient_records(all_records, test_ratio=test_ratio, seed=seed)
test_image_paths, test_mask_paths = records_to_paths(test_records)
print(f"test samples: {len(test_image_paths)}")
test_dataset = BUSIDataset(test_image_paths, test_mask_paths, image_transform=img_transform(), mask_transform=mask_transform(),
                           augmentation=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loss, test_dice, test_iou, test_acc, test_auc, results = evaluate_model_on_test(model_path="best_fold_1.pth", test_loader=test_loader, device=device)

results_sorted = sorted(results, key=lambda x: x["dice"], reverse=True)
show_prediction_overlay(results_sorted[:]) # best dice metrics cases
#show_prediction_overlay(results_sorted[-5:]) # worst dice metrics cases