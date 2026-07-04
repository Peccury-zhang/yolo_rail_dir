## 升级为“旋转目标检测”（OBB 方案，非常适合导轨）

如果你的导轨在画面中是**斜着**的（例如倾斜了 30° 或 45°），普通矩形框会引入大量的背景噪声，而且无法表达旋转角度。这时候你可以使用 YOLO 的 **OBB（Oriented Bounding Boxes，定向/旋转目标检测）**。

### 1. 什么是旋转目标检测？

它介于普通检测和多边形之间。它不是任意形状的多边形，而是一个**带角度的矩形（Rotated Rectangle）**。它的标签由 `(x_center, y_center, width, height, angle)` 组成。

### 2. 如何标注？

- 使用支持 OBB 的标注工具（如 **X-AnyLabeling**、**LabelStudio** 或 **Roboflow**）。
- 在标注时，你可以画一个可以旋转的矩形，让矩形的长边严格平行于导轨。

### 3. 在 YOLO 中如何训练？

如果你使用的是 YOLOv8 或 YOLOv11，它原生支持 OBB。

- **训练代码**极其简单，只需要加载带 `-obb` 后缀的模型即可：

Python

```
from ultralytics import YOLO

# 加载官方旋转目标检测模型
model = YOLO('yolov8n-obb.pt') 

# 训练
model.train(data='your_obb_data.yaml', epochs=100, imgsz=640)
```

## 🛒 针对你场景的最终选型建议