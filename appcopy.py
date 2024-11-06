from ultralytics import YOLOv10

# Load a model
model = YOLOv10("best.pt")  # load an official model
# Export the model
model.export(format="openvino",int8=True)
