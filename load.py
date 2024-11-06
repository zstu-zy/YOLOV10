import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import time
from openvino.runtime import Core

# 加载模型
model_xml = 'zybest.xml'
model_bin = 'zybest.bin'

# 类别名称映射
class_names = {0: "Header", 1: "Title", 2: "Text", 3: "Figure", 4: "Foot"}

# 创建 OpenVINO 核心对象
ie = Core()
model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model, "CPU")
input_layer = compiled_model.input(0)

# 输入图像文件夹和输出 XML 文件夹
input_folder = 'test'
output_folder = 'output'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 记录推理开始时间
start_time = time.time()

# 定义 NMS 函数
def non_max_suppression(boxes, scores, threshold=0.4):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        order = order[np.where(iou <= threshold)[0] + 1]

    return keep

# 遍历输入文件夹中的所有图像
for image_file in os.listdir(input_folder):
    if image_file.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_folder, image_file)

        # 读取输入图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像：{image_path}")
            continue

        # 预处理图像
        input_shape = input_layer.shape
        h, w = image.shape[:2]
        resize_w, resize_h = input_shape[3], input_shape[2]
        aspect_ratio = w / h
        new_w, new_h = (resize_w, int(resize_w / aspect_ratio)) if aspect_ratio > (resize_w / resize_h) else (int(resize_h * aspect_ratio), resize_h)
        image_resized = cv2.resize(image, (new_w, new_h))
        pad_w, pad_h = resize_w - new_w, resize_h - new_h
        top, bottom, left, right = pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2
        image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        image_padded = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)).astype(np.float32) / 255.0
        image_padded = image_padded[np.newaxis, :]

        # 执行推理
        results = compiled_model([image_padded])
        output_layer = compiled_model.output(0)
        boxes = results[output_layer][0]

        # 输出结果到 XML 文件
        confidence_threshold = 0.25
        root = ET.Element("detections")
        filtered_boxes = []
        filtered_scores = []

        for box in boxes:
            confidence = box[4]
            if confidence >= confidence_threshold:
                filtered_boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[5])])  # 添加 class_id
                filtered_scores.append(confidence)

        # 应用 NMS
        indices = non_max_suppression(filtered_boxes, filtered_scores, threshold=0.3)

        # 将检测结果写入 XML
        for i in indices:
            box = filtered_boxes[i]
            if len(box) < 5:  # 确保 box 至少包含 class_id
                print(f"警告：框数据格式错误，box: {box}")
                continue

            class_id = box[4]  # 获取类别 ID
            confidence = filtered_scores[i]

            # 从检测框坐标还原到原始图像坐标
            x_min = int((box[0] - left) / new_w * w)
            y_min = int((box[1] - top) / new_h * h)
            x_max = int((box[2] - left) / new_w * w)
            y_max = int((box[3] - top) / new_h * h)

            # 将检测结果写入 XML
            detection = ET.SubElement(root, "detection")
            detection.set("x_min", str(x_min))
            detection.set("y_min", str(y_min))
            detection.set("x_max", str(x_max))
            detection.set("y_max", str(y_max))
            detection.set("confidence", str(confidence))
            detection.set("class_id", str(class_id))

        # 保存 XML 文件
        xml_output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.xml")
        tree = ET.ElementTree(root)
        tree.write(xml_output_path)

        print(f"推理完成：{image_file}，结果已保存至 {xml_output_path}")

# 记录推理结束时间
end_time = time.time()
print(f"推理总耗时：{end_time - start_time:.2f}秒")
