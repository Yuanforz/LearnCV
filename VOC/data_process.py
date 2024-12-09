import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 定义文件夹路径
image_folder = 'TinySeg/JPEGImages'
mask_folder = 'TinySeg/Annotations'
output_folder = 'img'
output_txt = os.path.join(output_folder, 'labels.txt')

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取图像和掩码文件列表
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

# 颜色到类别的映射表
color_to_class = {}

# 打开输出TXT文件
with open(output_txt, 'w') as f:
    # 遍历图像和掩码文件
    for image_file, mask_file in tqdm(zip(image_files, mask_files)):
        # 读取图像和掩码
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        # 缩放图像到224x224
        image = image.resize((224, 224), Image.LANCZOS)

        # 保存缩放后的图像
        output_image_path = os.path.join(output_folder, image_file).replace('.jpg', '.png')
        image.save(output_image_path, 'PNG')

        # 统计掩码中包含的类别
        mask_np = np.array(mask)
        unique_colors = np.unique(mask_np.reshape(-1, mask_np.shape[2]), axis=0)
        classes = set()
        for color in unique_colors:
            color_tuple = tuple(color)
            if color_tuple != (0, 0, 0):  # 忽略背景色黑色
                if color_tuple not in color_to_class:
                    color_to_class[color_tuple] = len(color_to_class)
                classes.add(color_to_class[color_tuple])
        image_file = image_file.replace('.jpg', '')
        # 写入TXT文件
        if classes:
            f.write(f"{image_file},{','.join(map(str, sorted(classes)))}\n")

print("数据预处理完成！")