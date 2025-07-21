from ultralytics import YOLO

# load a model
model = YOLO("model.pt")

# Train the model

model.tune(data="dataset.yaml",
project="hyperparemeter_tuning",name="T1",
epochs=350,
iterations=45,
optimizer='SGD',
plots=False,
val=False,
batch=0.90,
box=8.0,
cls=1.0,
dfl=2.0,
hsv_h=0.03,
hsv_s=0.8,
hsv_v=0.5,
degrees=90.0,
translate=0.2,
scale=0.7,
shear=0.0,
perspective=0.0,
flipud=0.7,
fliplr=0.7,
bgr=0.6,
mosaic=1.0,
mixup=0.5,
lr0=0.01, momentum=0.9,weight_decay=0.0005,
warmup_epochs=3.0,
warmup_momentum=0.8,
warmup_bias_lr=0.1, cos_lr=True,dropout=0.0,patience=120
)