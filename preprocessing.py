import os
import random
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

def img_transform():
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    return transform

def mask_transform():
    msk_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    return msk_transform

def nlm_denoise(image, h=10, template_window_size=7, search_window_size=21):
    # reduce speckle noise with Non-Local Means (NLM)
    if image.dtype != np.uint8:
        image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_u8 = image
    return cv2.fastNlMeansDenoising(image_u8, None, h=h, templateWindowSize=template_window_size, searchWindowSize=search_window_size)

def rotate_180(image, mask):
    # rotate image and mask by 180 degrees
    return cv2.rotate(image, cv2.ROTATE_180), cv2.rotate(mask, cv2.ROTATE_180)

def train_augmentation(image, mask, use_nlm=True, use_rotation=True):
    if use_nlm:
        image = nlm_denoise(image)
    if use_rotation:
        image, mask = rotate_180(image, mask)
    return image, mask

def is_image_file(filename): # collect image-mask pairs
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))

def find_mask_paths(patient_path, image_stem):
    mask_paths = []
    prefix = f"{image_stem}_mask"
    for file_name in sorted(os.listdir(patient_path)):
        if not is_image_file(file_name):
            continue
        lower_name = file_name.lower()
        if lower_name.startswith(prefix):
            mask_paths.append(os.path.join(patient_path, file_name))
    return mask_paths

def get_patient_pairs(root_dir):
    """
    collect image/mask pairs grouped by patient and class.

    expected structure:
        root_dir/
            benign/
                patient1/
                    patient1.png
                    mask_patient1.png
                    ...
            malignant/
            normal/

    returns a list of records (each record is a dict with the keys:
    "class_name", "patient_id", "image_path", "mask_paths")
    """
    records = []
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        class_entries = [os.path.join(class_path, entry) for entry in sorted(os.listdir(class_path))]
        patient_dirs = [entry for entry in class_entries if os.path.isdir(entry)]

        if patient_dirs:
            for patient_path in patient_dirs:
                patient_id = os.path.basename(patient_path)
                for file_name in sorted(os.listdir(patient_path)):
                    if not is_image_file(file_name):
                        continue
                    if "_mask" in file_name.lower():
                        continue
                    image_path = os.path.join(patient_path, file_name)
                    stem, _ = os.path.splitext(file_name)
                    mask_paths = find_mask_paths(patient_path, stem)
                    if mask_paths:
                        records.append({
                            "class_name": class_name,
                            "patient_id": patient_id,
                            "image_path": image_path,
                            "mask_paths": mask_paths,
                        })
        else:
            for file_name in sorted(os.listdir(class_path)):
                if not is_image_file(file_name):
                    continue
                if "_mask" in file_name.lower():
                    continue
                image_path = os.path.join(class_path, file_name)
                stem, _ = os.path.splitext(file_name)
                mask_paths = find_mask_paths(class_path, stem)
                if mask_paths:
                    records.append({
                        "class_name": class_name,
                        "patient_id": stem,
                        "image_path": image_path,
                        "mask_paths": mask_paths,
                    })
    return records

def split_patient_pairs(root_dir, test_ratio=0.2, seed=42):
    # split to train & test set by patient, while keeping every class represented in both sets
    records = get_patient_pairs(root_dir)
    train_records, test_records = split_patient_records(records, test_ratio=test_ratio, seed=seed)
    train_image_paths = [r["image_path"] for r in train_records]
    train_mask_paths = [r["mask_paths"] for r in train_records]
    test_image_paths = [r["image_path"] for r in test_records]
    test_mask_paths = [r["mask_paths"] for r in test_records]
    return train_image_paths, train_mask_paths, test_image_paths, test_mask_paths

def split_patient_records(records, test_ratio=0.2, seed=42):
    # split records to train & test by patient, while keeping every class represented in both sets
    by_class = {}
    for record in records:
        by_class.setdefault(record["class_name"], {}).setdefault(record["patient_id"], []).append(record)
    rng = random.Random(seed)
    train_records = []
    test_records = []
    for class_name, patients in by_class.items():
        patient_ids = list(patients.keys())
        rng.shuffle(patient_ids)
        if len(patient_ids) <= 1:
            for pid in patient_ids:
                train_records.extend(patients[pid])
            continue
        n_test = max(1, int(round(len(patient_ids) * test_ratio)))
        n_test = min(n_test, len(patient_ids) - 1)
        test_patient_ids = set(patient_ids[:n_test])
        for pid, samples in patients.items():
            if pid in test_patient_ids:
                test_records.extend(samples)
            else:
                train_records.extend(samples)
    return train_records, test_records

def get_busi_pairs(root_dir):
    records = get_patient_pairs(root_dir)
    image_paths = [r["image_path"] for r in records]
    mask_paths = [r["mask_paths"] for r in records]
    return image_paths, mask_paths
#load one image-mask pair at a time
class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_transform=None, mask_transform=None, augmentation=None):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augmentation = augmentation
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = image.astype("float32")
        mask_files = self.mask_paths[idx]
        if len(mask_files) == 0:
            raise ValueError(f"no mask files found for image: {self.image_paths[idx]}")
        mask = None
        for mask_file in mask_files:
            current_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if current_mask is None:
                continue
            current_mask = (current_mask > 0).astype("float32")
            if mask is None:
                mask = current_mask
            else:
                if current_mask.shape != mask.shape:
                    current_mask = cv2.resize(current_mask, (mask.shape[1], mask.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                mask = ((mask > 0) | (current_mask > 0)).astype("float32")
        if mask is None:
            raise ValueError(f"couldn't read any mask files for image: {self.image_paths[idx]}")

        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return image, mask
