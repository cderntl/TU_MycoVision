import sys
import pandas as pd
import re
from io import StringIO
from ultralytics import YOLO

# List of models to validate
model_paths = [
"path1",
"path2",

]

# Dataset path
dataset_path = "/dataset.yaml"

# Store results in a list of dictionaries
all_results = []

# Iterate through models
for model_path in model_paths:
    print(f"Validating {model_path}...")

    # Load model
    model = YOLO(model_path)

    # Run validation
    model.val(data=dataset_path, split="val")
 