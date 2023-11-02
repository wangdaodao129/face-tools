import os
import cv2
from face_alignment import FaceAlignment, LandmarksType
import numpy as np

# 初始化FaceAlignment模型
fa = FaceAlignment(LandmarksType.TWO_HALF_D, device='cpu')

# 输入图片路径和输出路径
input_dir = "data/input"  # 输入图片文件夹路径
output_dir = "data/output"  # 输出图片文件夹路径

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 16:9
high_r = 16
width_r = 9

# 遍历输入文件夹中的所有图片文件
for filename in os.listdir(input_dir):

    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 读取图片
        image = cv2.imread(input_path)
        H, W, _ = image.shape

        # 进行人脸检测
        landmarks = fa.get_landmarks(image)

        if landmarks is not None:
            # 获取人脸的中心坐标
            center_h = int(np.mean(landmarks[0][:, 1]))
            center_w = int(np.mean(landmarks[0][:, 0]))
            h = H
            w = W
            left = 0
            right = w
            top = 0
            bottom = h

            # h :w = 16:9
            if (H/W) < (high_r/width_r):
                # 太宽
                new_w = H * width_r / high_r
                w = new_w
                left = center_w - w/2
                right = center_w + w/2
                if left < 0:
                    p = 0 - left
                    left = 0
                    right += p
                elif right > W:
                    p = right - W
                    right = W
                    left -= p
            elif (H/W) > (high_r/width_r):
                # 太高
                new_h = W * high_r / width_r
                h = new_h
                top = center_h - h/2
                bottom = center_h + h/2
                if top < 0:
                    p = 0 - top
                    top = 0
                    bottom += p
                elif bottom > H:
                    p = bottom - H
                    bottom = H
                    top -= p

            # 裁剪
            print(f'{top}, {bottom}, {left}, {right}')
            cropped_face = image[int(top): int(bottom), int(left): int(right)]

            # 保存裁剪后的人脸图片
            cv2.imwrite(output_path, cropped_face)
