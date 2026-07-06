# -*- coding: utf-8 -*-
"""
按类别比例划分数据集:
- 从 labels_trans + images 中, 按各类别比例抽取共 20 个样本作为验证集(valid_data)
- 其余样本作为训练集(train_data)
- 图片放入 <split>/images, 标签放入 <split>/labels
使用固定随机种子, 结果可复现。
"""
import os
import shutil
import random
from collections import defaultdict, Counter

random.seed(42)

ROOT = os.path.dirname(__file__)
LABELS_DIR = os.path.join(ROOT, 'labels_trans')
IMAGES_DIR = os.path.join(ROOT, 'images')
TRAIN_DIR = os.path.join(ROOT, 'train_model', 'train_data')
VALID_DIR = os.path.join(ROOT, 'train_model', 'valid_data')

CLASSES = ['up', 'down', 'left', 'right']
VALID_TOTAL = 20
IMG_EXT = '.png'

# 1. 按主类别对文件分组
groups = defaultdict(list)
for txt in os.listdir(LABELS_DIR):
    if not txt.endswith('.txt'):
        continue
    path = os.path.join(LABELS_DIR, txt)
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        continue
    classes_in_file = [int(ln.split()[0]) for ln in lines]
    main_cls = Counter(classes_in_file).most_common(1)[0][0]
    stem = os.path.splitext(txt)[0]
    groups[main_cls].append(stem)

counts = {c: len(groups[c]) for c in range(len(CLASSES))}
total = sum(counts.values())

# 2. 最大余数法计算每个类别的验证集配额
raw = {c: VALID_TOTAL * counts[c] / total for c in counts}
alloc = {c: int(raw[c]) for c in counts}
remainder = VALID_TOTAL - sum(alloc.values())
# 按小数余数从大到小分配剩余名额, 平局按类别索引
order = sorted(counts, key=lambda c: (raw[c] - int(raw[c]), -c), reverse=True)
for c in order[:remainder]:
    alloc[c] += 1

print("类别文件数:", {CLASSES[c]: counts[c] for c in counts})
print("验证集配额:", {CLASSES[c]: alloc[c] for c in counts}, "合计", sum(alloc.values()))

# 3. 抽取验证集样本
valid_stems, train_stems = [], []
for c in range(len(CLASSES)):
    files = sorted(groups[c], key=lambda x: int(x) if x.isdigit() else x)
    random.shuffle(files)
    valid_stems.extend(files[:alloc[c]])
    train_stems.extend(files[alloc[c]:])


def prepare(split_dir):
    img_dir = os.path.join(split_dir, 'images')
    lbl_dir = os.path.join(split_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    return img_dir, lbl_dir


def copy_set(stems, split_dir):
    img_dir, lbl_dir = prepare(split_dir)
    ok = 0
    for stem in stems:
        src_img = os.path.join(IMAGES_DIR, stem + IMG_EXT)
        src_lbl = os.path.join(LABELS_DIR, stem + '.txt')
        if not os.path.exists(src_img):
            print(f"  警告: 缺少图片 {src_img}, 跳过")
            continue
        shutil.copy2(src_img, os.path.join(img_dir, stem + IMG_EXT))
        shutil.copy2(src_lbl, os.path.join(lbl_dir, stem + '.txt'))
        ok += 1
    return ok


n_valid = copy_set(valid_stems, VALID_DIR)
n_train = copy_set(train_stems, TRAIN_DIR)

print(f"验证集: {n_valid} 个 -> {VALID_DIR}")
print(f"训练集: {n_train} 个 -> {TRAIN_DIR}")
print("完成。")
