from ultralytics import YOLO
import os

data_path = r"C:\Users\zhangye\Desktop\yolov10-main\datasets\my_data.yaml"
model = YOLO(r'C:\Users\zhangye\Desktop\yolov10-main\zybest.pt')

# 继续您的导出代码
model.export(format="openvino", int8=True, data=data_path)
