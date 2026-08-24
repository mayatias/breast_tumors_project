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
                patient1.png
                patient1_mask.png
                patient2.png
                patient2_mask.png
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

#load one image-mask pair at a time
class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_transform=None, mask_transform=None, augmentation=False):
        assert len(image_paths) == len(mask_paths)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augmentation = augmentation
        self.samples = [] # save all the samples
        self.prepare_dataset(image_paths, mask_paths)
    def prepare_dataset(self, image_paths, mask_paths):
        for image_path, mask_files in zip(image_paths, mask_paths):
            # load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"couldn't read image: {image_path}")
            image = image.astype("float32")
            # load masks
            if len(mask_files) == 0:
                raise ValueError(f"no mask files found for image: {image_path}")
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
                        current_mask = cv2.resize(current_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask = ((mask > 0) | (current_mask > 0)).astype("float32")
            if mask is None:
                raise ValueError(f"couldn't read any mask for image: {image_path}")
            img = image
            msk = mask
            if self.image_transform:
                img = self.image_transform(img)
            if self.mask_transform:
                msk = self.mask_transform(msk)
            msk = (msk > 0.5).float()
            self.samples.append((img, msk))
            # augmentations
            if self.augmentation:
                # NLM
                img_nlm = nlm_denoise(image)
                msk_nlm = mask
                if self.image_transform:
                    img_nlm = self.image_transform(img_nlm)
                if self.mask_transform:
                    msk_nlm = self.mask_transform(msk_nlm)
                msk_nlm = (msk_nlm > 0.5).float()
                self.samples.append((img_nlm, msk_nlm))
                # rotation
                img_rot, msk_rot = rotate_180(image, mask)
                if self.image_transform:
                    img_rot = self.image_transform(img_rot)
                if self.mask_transform:
                    msk_rot = self.mask_transform(msk_rot)
                msk_rot = (msk_rot > 0.5).float()
                self.samples.append((img_rot, msk_rot))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        image, mask = self.samples[idx]

        return image, mask