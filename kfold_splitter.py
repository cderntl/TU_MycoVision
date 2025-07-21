import yaml
from pathlib import Path
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import datetime
import shutil

# Path to the dataset
dataset_path = Path("path")  # replace with 'path/to/dataset' for your custom data

# Gather label files
labels = list(sorted(dataset_path.rglob("*labels/*.txt")))  # all data in 'labels'

# Load class names from YAML file
yaml_file ="/dataset.yaml"  # your data YAML with data directories and names dictionary
with open(yaml_file, "r", encoding="utf8") as y:
    classes = yaml.safe_load(y)["names"]

# Create a list of indices based on the length of the class names list
cls_idx = list(range(len(classes)))

# Create a list of base filenames (without extensions) from labels
indx = [l.stem for l in labels]  # uses base filename as ID (no extension)

# Initialize an empty pandas DataFrame
labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

# Count the instances of each class-label present in the annotation files
for label in labels:
    lbl_counter = Counter()
    with open(label, "r") as lf:
        lines = lf.readlines()
    for l in lines:
        # classes for YOLO label uses integer at first position of each line
        lbl_counter[int(l.split(" ")[0])] += 1
    # Use .loc to avoid the SettingWithCopyWarning
    labels_df.loc[label.stem] = lbl_counter

labels_df = labels_df.fillna(0.0).infer_objects(copy=False)  # replace `nan` values with `0.0` and infer types

# Output the first few rows of the DataFrame to verify
print(labels_df.head())

# Initialize StratifiedKFold cross-validation
ksplit = 5
skf = StratifiedKFold(n_splits=ksplit, shuffle=True, random_state=20)

# Generate stratified splits
kfolds = list(skf.split(labels_df, labels_df.idxmax(axis=1)))

# Construct a DataFrame to display train/val splits
folds = [f"split_{n}" for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=indx, columns=folds)

for idx, (train, val) in enumerate(kfolds, start=1):
    folds_df.loc[labels_df.iloc[train].index, f"split_{idx}"] = "train"
    folds_df.loc[labels_df.iloc[val].index, f"split_{idx}"] = "val"

# Calculate and store label distribution ratios for each fold
fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio

# Initialize an empty list to store image file paths
supported_extensions = [".jpg", ".jpeg", ".png"]
images = []

# Loop through supported extensions and gather image files
for ext in supported_extensions:
    images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

# Create the necessary directories and dataset YAML files
save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
save_path.mkdir(parents=True, exist_ok=True)
ds_yamls = []

for split in folds_df.columns:
    # Create directories
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Create dataset YAML files
    dataset_yaml = split_dir / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)

    with open(dataset_yaml, "w") as ds_y:
        yaml.safe_dump(
            {
                "path": split_dir.as_posix(),
                "train": "train",
                "val": "val",
                "nc": len(classes),
                "names": classes,
            },
            ds_y,
        )

# Copy images and labels into the respective directory ('train' or 'val') for each split
for image, label in zip(images, labels):
    for split, k_split in folds_df.loc[image.stem].items():
        # Destination directory
        img_to_path = save_path / split / k_split / "images"
        lbl_to_path = save_path / split / k_split / "labels"

        # Copy image and label files to new directory (SamefileError if file already exists)
        shutil.copy(image, img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)

# Save DataFrames to CSV files
folds_df.to_csv(save_path / "kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")
