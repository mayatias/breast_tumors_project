import random
from collections import defaultdict

def records_to_paths(records):
    image_paths = [r["image_path"] for r in records]
    mask_paths = [r["mask_paths"] for r in records]
    return image_paths, mask_paths

def make_stratified_patient_folds(records, n_splits, seed=42):
    by_class_and_patient = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class_and_patient[record["class_name"]][record["patient_id"]].append(record)
    folds = [[] for _ in range(n_splits)]
    fold_class_patient_counts = [defaultdict(int) for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]
    rng = random.Random(seed)
    for class_name in sorted(by_class_and_patient.keys()):
        patient_items = list(by_class_and_patient[class_name].items())
        rng.shuffle(patient_items)
        patient_items.sort(key=lambda item: len(item[1]), reverse=True)
        for patient_id, patient_records in patient_items:
            target_fold = min(range(n_splits), key=lambda i: (fold_class_patient_counts[i][class_name], fold_sizes[i]))
            folds[target_fold].extend(patient_records)
            fold_class_patient_counts[target_fold][class_name] += 1
            fold_sizes[target_fold] += len(patient_records)
    return folds