# TU_MyCo-Vision: A Deep Learning Tool for Detection of Cell Morphologies in Fungal Microscopic Images

- TU_MyCo-Vision is an Ultralytics YOLOv11m-powered fungal cell detection and analysis tool.
- Designed to detect 13 different fungal cell morphologies in bright-field microscopic images.
- The tool is an end-to-end solution that integrates: Detection - Data Analysis - Visualization.

## Key Features
- Training & Validation dataset of 1504 images, comprising *Aureobasidium pullulans*, *A. melanogenum*, and *Trichoderma reesei*.
- Cross-species compatibility (Tested on images of *Komagataella phaffii* strains, *Candida albicans*, and *Aspergillus niger* spores).
- Class definitions untied from species-specific or cultivation condition-specific phenotypes, enabling wider compatibility with different research questions.
- Graphical User Interface for non-programmers, which combines:
  - Image detection.
  - Quantitative data analysis that supports:
    - Analysis of all images within a single sample group (Single-Group Analysis).
    - Analysis across multiple sample groups (Multi-Group Analysis).
    - Interactive plots generated using Plotly (v6.1.2).
  - Result Preview:
    - Paired preview of original and annotated images for qualitative analysis.

## Scripts
1. `cropper.py`  
   Used for generating 640 × 640 crops from 1920 × 1080 images while implementing a dataset expansion strategy.
2. `annotation_handler.py`  
   Counts the number of objects belonging to each label in a dataset with annotations in JSON format.
3. `kfold_splitter.py`  
   Used to split each dataset family into 5 train and validation splits. Base code adapted from Ultralytics (https://docs.ultralytics.com/guides/kfold-cross-validation/).
4. `hyperparameter_tuning.py`  
   Script used to perform hyperparameter search using the `model.tune()` method in Ultralytics. Base code adapted from Ultralytics (https://docs.ultralytics.com/guides/hyperparameter-tuning/).
5. `cross_validation_training.py`
   Script used to perfom the training with cross-validation on all 5 dataset splits.
7. `datafeed_prediction.py`  
   Script used for running the prediction function in the Ultralytics YOLO framework for auto-labelling operations.
8. `all_validator.py`  
   Script used for validating all MyCo-Vision model families on the global test dataset.

## Dependencies and Usage
To set up environment for using the tool with "TU_MyCo-Vision_v1_source_code.py". 
Use command: conda env create -f MyCo-Vision_environment.yaml --name name of your choice
All required packages are listed in the "MyCo-Vision_environment.yaml".

## Credits and Attributions

See the credits.md for list of all third-party packages and cited libraries used in this project.

## Acknowledgements
Developed as a part of study on polymorphism in *Aureobasidium pullulans*. Special thanks to all the contributors. This research was funded in whole or in part by the Austrian Science Fund (FWF) [10.55776/P 35642]. For open access purposes, the author has applied a CC BY public copyright license.
