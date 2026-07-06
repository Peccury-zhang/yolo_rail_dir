"""
分割数据集的json转txt脚本文件
"""
# -*- coding: utf-8 -*-
import json
import os
from tqdm import tqdm


def convert_label_json(json_dir, save_dir, classes):
    class_dict = {cls: idx for idx, cls in enumerate(classes.split(','))}
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    for json_file in tqdm(json_files):
        json_path = os.path.join(json_dir, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON file: {json_path}")
            continue

        image_height = data.get('imageHeight', None)
        image_width = data.get('imageWidth', None)
        if image_height is None or image_width is None:
            print(f"Image height or width not found in JSON file: {json_path}")
            continue

        shapes = data.get('shapes', [])
        if not shapes:
            print(f"No shapes found in JSON file: {json_path}")
            continue

        txt_file_path = os.path.join(save_dir, json_file.replace('.json', '.txt'))
        with open(txt_file_path, 'w') as txt_f:
            for shape in shapes:
                label = shape.get('label', None)
                points = shape.get('points', [])
                if label is None or not points:
                    print(f"Invalid shape entry in JSON file: {json_path}")
                    continue

                if label not in class_dict:
                    print(f"Label '{label}' not found in classes list. Skipping...")
                    continue

                label_index = class_dict[label]

                # OBB 要求每个目标为 4 个角点（8 个坐标）
                if len(points) != 4:
                    print(f"Shape '{label}' has {len(points)} points (expected 4) in JSON file: {json_path}. Skipping...")
                    continue

                normalized_points = []
                for point in points:
                    if len(point) != 2:
                        print(f"Invalid point {point} in JSON file: {json_path}")
                        continue
                    # 归一化并裁剪到 [0, 1]，避免旋转角点越界
                    normalized_x = min(max(point[0] / image_width, 0.0), 1.0)
                    normalized_y = min(max(point[1] / image_height, 0.0), 1.0)
                    normalized_points.extend([normalized_x, normalized_y])

                if len(normalized_points) != 8:
                    print(f"No valid 4-point box for label '{label}' in JSON file: {json_path}")
                    continue

                normalized_points_str = ' '.join(map(str, normalized_points))
                txt_line = f"{label_index} {normalized_points_str}\n"
                txt_f.write(txt_line)


# 直接设置参数
json_dir = 'C:/Users/15401/Desktop/yolo_rail_dir/labels'  # 替换为你的JSON文件目录
save_dir = 'C:/Users/15401/Desktop/yolo_rail_dir/labels_trans'  # 替换为你想要保存TXT文件的目录
classes = 'up,down,left,right'  # 替换为你的类别名称，用逗号分隔

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 执行转换
convert_label_json(json_dir, save_dir, classes)


