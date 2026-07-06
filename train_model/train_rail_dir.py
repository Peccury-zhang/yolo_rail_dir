# -*- coding: utf-8 -*-
"""
YOLO OBB(旋转目标检测) 训练脚本 —— 导轨方向检测(up/down/left/right)
依赖: pip install ultralytics
运行: python train_rail_dir.py
"""
import os
from ultralytics import YOLO

# 当前脚本所在目录 (train_model/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, 'data.yaml')


def main():
    # 加载官方 OBB 预训练模型 (n=最小最快; 可换 s/m/l/x 提升精度)
    # 首次运行会自动下载 yolov8n-obb.pt
    model = YOLO('yolov8n-obb.pt')

    model.train(
        data=DATA_YAML,
        epochs=200,
        imgsz=640,
        batch=8,
        device=0,            # 有 GPU 用 0; 纯 CPU 改成 'cpu'
        workers=4,
        patience=50,         # 50 轮无提升则早停
        project=os.path.join(BASE_DIR, 'runs'),
        name='rail_obb',
        exist_ok=True,
        # ---- 数据增强(针对方向检测, 关闭会改变方向语义的翻转) ----
        fliplr=0.0,          # 关闭左右翻转(会把 left/right 弄反)
        flipud=0.0,          # 关闭上下翻转(会把 up/down 弄反)
        degrees=0.0,         # 关闭随机旋转(会破坏方向标签)
        mosaic=1.0,
        close_mosaic=10,     # 最后 10 轮关闭 mosaic 以稳定收敛
    )

    # 在验证集上评估最佳权重
    metrics = model.val()
    print('验证结果 mAP50-95(OBB):', metrics.box.map)
    print('验证结果 mAP50(OBB):', metrics.box.map50)


if __name__ == '__main__':
    main()
