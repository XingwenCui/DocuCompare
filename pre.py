import math
import cv2
import numpy as np
import json


def calculate_angle(point1, point2):
    """
    计算两点构成的斜线与水平线的夹角。
    参数:
    point1 (list): 第一个点的坐标 [x1, y1]
    point2 (list): 第二个点的坐标 [x2, y2]
    返回:
    float: 两点之间的角度，正值表示第二个点在第一个点上方，负值表示在下方。
    """
    # 计算水平和垂直距离
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    # 使用atan2计算角度（返回值是弧度）
    angle_rad = math.atan2(dy, dx)
    # 将弧度转换为度
    angle_deg = math.degrees(angle_rad)
    # 根据y坐标判断返回正角度还是负角度
    if dy < 0:
        # 第二个点在第一个点的下方
        return -abs(angle_deg)
    else:
        # 第二个点在第一个点的上方或同一水平线上
        return abs(angle_deg)


def rotate_image_without_cropping(img, angle, scale=1.0):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    # 计算旋转矩阵（考虑不裁剪的情况）
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 计算旋转后图像的新边界
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # 新的边界尺寸
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    # 旋转整个图像
    rotated = cv2.warpAffine(img, M, (nW, nH))
    return rotated


def rotate_image(img, angle):
    # 获取图像维度和中心点
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 旋转中心，旋转角度，缩放因子
    # 执行旋转
    rotated_img = cv2.warpAffine(img, M, (w, h))

    return rotated_img


if __name__ == "__main__":

    with open("20250813110133_6_res.json", 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    results = load_dict['dt_polys']

    filename = '20250813110133_6.jpg'


    img = cv2.imread(filename)

    # 获取图片宽高
    height, width, = img.shape[:2]


    angleTotal = 0
    angleNum = 2*len(results)
    # img_rect=img
    for result in results:
        angleTotal += calculate_angle(result[0],result[1])
        angleTotal += calculate_angle(result[3], result[2])

    angle = angleTotal / (angleNum)
    img = rotate_image(img, angle)

    # cv2.imwrite(r"./rect/" +filename, img_rect)
    cv2.imwrite('rotate2.png', img)

    print(angle)




