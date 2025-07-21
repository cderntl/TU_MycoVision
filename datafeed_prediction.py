from ultralytics import YOLO

# Load a model
model = YOLO("model.pt")  # pretrained YOLO11n model

# Define path to directory 
source="path"

#Inference
model.predict(source=source, iou=0.5, imgsz=(640,640), augment=True, agnostic_nms=True, visualize=False,
save=True, name="name, show=False, save_txt=True, line_width=1, save_crop=False, conf=0.2,device=0)
