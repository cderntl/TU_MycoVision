from ultralytics import YOLO

results = {}

# List of dataset YAML files (paths to your 5 dataset splits)
ds_yamls = [
    "/split_1_dataset.yaml",
    "/split_2_dataset.yaml",
    "/split_3_dataset.yaml",
    "/split_4_dataset.yaml",
    "/split_5_dataset.yaml"
]

#Number of splits
ksplit = len(ds_yamls)

# Iterate over each dataset YAML file to run training
for k in range(ksplit):
    # Re-initialize the model inside the loop to ensure a fresh start for each fold
    model = YOLO("yolo11m.pt", task="detect")
    
    #get current dataset YAML file

    dataset_yaml = ds_yamls[k]
    
    # Train the model with specified arguments
    model.train(
        data=dataset_yaml,
epochs=1000,
patience=130,
batch=0.90,
cos_lr=True,
plots=True, 
optimizer='SGD',
device=0,
lr0=0.01551,
lrf=0.01106,
momentum=0.9215,
weight_decay=0.00053,
warmup_epochs=2.76134,
warmup_momentum=0.77505,
box=8.85889,
cls=0.97851,
dfl=1.9413,
hsv_h=0.03244,
hsv_s=0.65832,
hsv_v=0.44335,
degrees=44.66121,
translate=0.23812,
scale=0.61362,
shear=0.0,
perspective=0.0,
flipud=0.7841,
fliplr=0.70311,
bgr=0.51936,
mosaic=0.56641,
mixup=0.41302,
copy_paste=0.0,
project="name")
    
results[k] = model.metrics
