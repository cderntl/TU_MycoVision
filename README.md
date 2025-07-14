# TU_MyCo-Vision: A Deep Learning Tool for Detection of Cell Morphologies in Fungal Microscopic Images

- TU_MyCo-Vision is an Ultralytics YOLOv11m powered fungal cell detection and analysis tool 
- Designed to detect 13 different fungal cell morphologies in bright-field microscopic images  
- The tool is an end-to-end solution which integrates: Detection - Data Analysis - Visualization

## Key Features
- Dataset of 1504 images, comprising of *Aureobasidium pullulans*, *A.melanogenum* and *Trichoderma ressei*
- Cross-species compatibility (Tested on images of *Komagataella phaffi* strains, *Candida albicans* & *Aspergillus niger* spores)
- Class definitions un-tied with species-specied or cultivation condition specific phenotypes, enabling wider compatibility with different researarch question
- Graphical User Interface for non-programmers which combines
    - Image detection
    - Quantitative data analysis which supports:
        - Analysis of all images within a single sample group (Single-Group Analysis)
        - Analysis across multiple sample groups (Multi-Group Analysis)
        - Interactive plots generated using plotly(v 6.1.2)
    - Result Preview
        - Paired preview of original and annotated image for qualitative analysis
## Acknowledgements 
Deveoped as a part of study on polymorphism in *Aureobasidium pullulans*. Special thanks to all the contributors.
