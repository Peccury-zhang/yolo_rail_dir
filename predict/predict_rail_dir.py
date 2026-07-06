# -*- coding: utf-8 -*-
"""
YOLO OBB(旋转目标检测) 推理脚本 —— 导轨方向检测(up/down/left/right)
使用 best.pt 对 source_img/ 中的图片进行推理, 输出带旋转框和 label 的结果到 output_img/。
依赖: pip install ultralytics
运行: python predict_rail_dir.py
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

# 当前脚本所在目录 (predict/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
SOURCE_DIR = os.path.join(BASE_DIR, 'source_img')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_img')

# 支持的图片格式
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# 每个类别一个固定颜色 (BGR)
COLORS = {
    'up': (0, 0, 255),      # 红
    'down': (0, 255, 0),    # 绿
    'left': (255, 0, 0),    # 蓝
    'right': (0, 255, 255), # 黄
}
DEFAULT_COLOR = (255, 255, 255)


def draw_obb(img, poly, label, conf, color):
    """在图像上绘制单个旋转框和标签。poly 为 4x2 的角点坐标。"""
    pts = poly.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    text = f'{label} {conf:.2f}'
    # 以第一个角点作为文字锚点
    x, y = int(poly[0][0]), int(poly[0][1])
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(y, th + 5)
    cv2.rectangle(img, (x, y_text - th - baseline), (x + tw, y_text + baseline), color, -1)
    cv2.putText(img, text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'找不到模型: {MODEL_PATH}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)
    names = model.names  # {index: name}

    # 收集待推理图片
    files = [f for f in os.listdir(SOURCE_DIR)
             if os.path.splitext(f)[1].lower() in IMG_EXTS]
    if not files:
        print(f'source_img 目录为空: {SOURCE_DIR}')
        return

    print(f'共发现 {len(files)} 张图片, 开始推理...')

    for fname in files:
        src_path = os.path.join(SOURCE_DIR, fname)
        img = cv2.imread(src_path)
        if img is None:
            print(f'[跳过] 无法读取图片: {fname}')
            continue

        results = model.predict(source=src_path, verbose=False)
        r = results[0]

        n = 0
        if r.obb is not None and len(r.obb) > 0:
            polys = r.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
            clss = r.obb.cls.cpu().numpy().astype(int)
            confs = r.obb.conf.cpu().numpy()
            n = len(polys)
            for poly, c, conf in zip(polys, clss, confs):
                label = names.get(int(c), str(c))
                color = COLORS.get(label, DEFAULT_COLOR)
                draw_obb(img, poly, label, float(conf), color)

        out_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, img)
        print(f'[完成] {fname} -> 检测到 {n} 个目标, 已保存到 output_img/')

    print(f'全部完成! 结果已保存到: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
