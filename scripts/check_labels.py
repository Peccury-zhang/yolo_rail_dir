# -*- coding: utf-8 -*-
"""
校验 labels_trans 中的 YOLO-OBB 标签是否合法，并统计每个类别的实例数量与文件数量。
OBB 每行格式: class_index x1 y1 x2 y2 x3 y3 x4 y4  (9 个字段, 坐标归一化到 [0,1])
"""
import os
from collections import Counter

LABELS_DIR = os.path.join(os.path.dirname(__file__), 'labels_trans')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')
CLASSES = ['up', 'down', 'left', 'right']

instance_counter = Counter()   # 每个类别的目标(框)数量
file_class_counter = Counter()  # 每个类别出现在多少个文件中(以文件首个/主要类别)
errors = []
empty_files = []
multi_shape_files = []

txt_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]

for txt in sorted(txt_files):
    path = os.path.join(LABELS_DIR, txt)
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        empty_files.append(txt)
        continue

    if len(lines) > 1:
        multi_shape_files.append((txt, len(lines)))

    file_classes = []
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 9:
            errors.append(f"{txt} 第{i+1}行: 字段数={len(parts)} (应为9)")
            continue
        try:
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
        except ValueError:
            errors.append(f"{txt} 第{i+1}行: 存在无法解析的数值")
            continue
        if cls < 0 or cls >= len(CLASSES):
            errors.append(f"{txt} 第{i+1}行: 类别索引 {cls} 越界")
            continue
        out_of_range = [c for c in coords if c < 0.0 or c > 1.0]
        if out_of_range:
            errors.append(f"{txt} 第{i+1}行: 坐标越界 {out_of_range}")
        instance_counter[cls] += 1
        file_classes.append(cls)

    # 记录该文件的主类别(取出现最多的类别)
    if file_classes:
        main_cls = Counter(file_classes).most_common(1)[0][0]
        file_class_counter[main_cls] += 1

# 检查图片与标签是否一一对应
img_stems = {os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR)}
lbl_stems = {os.path.splitext(f)[0] for f in txt_files}
missing_img = sorted(lbl_stems - img_stems, key=lambda x: int(x) if x.isdigit() else x)
missing_lbl = sorted(img_stems - lbl_stems, key=lambda x: int(x) if x.isdigit() else x)

print("=" * 50)
print(f"标签文件总数: {len(txt_files)}")
print(f"图片文件总数: {len(img_stems)}")
print("-" * 50)
print("每个类别的【目标框】数量(instances):")
for i, name in enumerate(CLASSES):
    print(f"  {i} {name:5s}: {instance_counter[i]}")
print(f"  合计: {sum(instance_counter.values())}")
print("-" * 50)
print("每个类别的【文件】数量(按文件主类别):")
for i, name in enumerate(CLASSES):
    print(f"  {i} {name:5s}: {file_class_counter[i]}")
print(f"  合计: {sum(file_class_counter.values())}")
print("-" * 50)
print(f"空标签文件: {len(empty_files)} -> {empty_files}")
print(f"含多个目标的文件: {len(multi_shape_files)} -> {multi_shape_files}")
print(f"有标签但缺图片: {len(missing_img)} -> {missing_img}")
print(f"有图片但缺标签: {len(missing_lbl)} -> {missing_lbl}")
print("-" * 50)
if errors:
    print(f"发现 {len(errors)} 处格式错误:")
    for e in errors:
        print("  " + e)
else:
    print("格式校验: 全部通过 ✔ (每行9字段, 坐标均在[0,1])")
print("=" * 50)
