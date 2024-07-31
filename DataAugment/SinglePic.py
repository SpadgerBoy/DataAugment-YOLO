'''
单图片数据增强
翻转，HSV增强、旋转、平移、缩放、剪切、透视、
'''

import math
import os
import random
import cv2
import numpy as np
import shutil
from math import cos, sin
from PIL import Image


# 计算结果保留6位小数
def roundxywh(x, y, w, h):
    x = round(x, 6)
    y = round(y, 6)
    w = round(w, 6)
    h = round(h, 6)
    return x, y, w, h


# 处理标签数据越界问题
def boundary(xc,yc,w,h):
    if xc-w/2 < 0.0 or yc-h/2 < 0.0 or xc+w/2 > 1.0 or yc + h/2 > 1.0:
        return 0
    else:
        return 1


# 坐标旋转计算函数——计算旋转后的坐标
def rotate_xy(x, y, angle, cx, cy):  # 参数需要旋转点的x、y坐标，旋转角度、旋转中心点的x、y坐标
    # 点(x,y) 绕(cx,cy)点旋转
    angle = angle * np.pi / 180
    x_new = float((x - cx) * cos(angle) - (y - cy) * sin(angle) + cx)
    y_new = float((x - cx) * sin(angle) + (y - cy) * cos(angle) + cy)
    return x_new, y_new



# 翻转、HSV增强、旋转、平移、缩放、剪切、透视
def data_augment(img_path, oldLabel, newLabel, method):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    label = open(oldLabel)  # 读标签
    if os.path.exists(newLabel):
        os.remove(newLabel)
    fb = open(newLabel, mode='a', encoding='utf-8')  # 创建一个新的txt的文本用于存储新的标签

    # 左右水平翻转
    if method == 'fliplr':
        # 变换图片
        img = np.fliplr(img)
        # 变换标签
        print(label)
        lines = label.readlines()  # 读出label中的每一行
        for line in lines:  # 把lines中的数据逐行读取出来
            list = line.strip('\n').split(' ')
            nxc = round(1.0 - float(list[1]), 6)
            newList = list[0] + ' ' + str(nxc) + ' ' + list[2] + ' ' + list[3] + ' ' + list[4] + '\n'
            fb.write(newList)

    # 上下垂直翻转
    elif method == 'flipud':
        # 变换图片
        img = np.flipud(img)
        # 变换标签
        lines = label.readlines()  # 读出label中的每一行
        for line in lines:  # 把lines中的数据逐行读取出来
            list = line.strip('\n').split(' ')
            nyc = round(1.0 - float(list[2]), 6)
            newList = list[0] + ' ' + list[1] + ' ' + str(nyc) + ' ' + list[3] + ' ' + list[4] + '\n'
            fb.write(newList)

    # hsv色域变换
    elif method == 'hsv':
        """hsv色域增强  处理图像hsv，不对label进行任何处理
        :param img: 待处理图片  BGR [736, 736]
        :param hgain: h通道色域参数 用于生成新的h通道
        :param sgain: h通道色域参数 用于生成新的s通道
        :param vgain: h通道色域参数 用于生成新的v通道
        :return: 返回hsv增强后的图片 img
        """
        # 变换图片
        hgain, sgain, vgain = 0.015, 0.7, 0.4
        if hgain or sgain or vgain:
            # 随机取-1到1三个实数，乘以hyp中的hsv三通道的系数  用于生成新的hsv通道
            r = np.random.uniform(-2, 3, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))  # 图像的通道拆分 h s v
            dtype = img.dtype  # uint8

            # 构建查找表
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)  # 生成新的h通道
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 生成新的s通道
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 生成新的v通道

            # 图像的通道合并 img_hsv=h+s+v  随机调整hsv之后重新组合hsv通道
            # cv2.LUT(hue, lut_hue)   通道色域变换 输入变换前通道hue 和变换后通道lut_hue
            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        # 变换标签
        lines = label.readlines()  # 读出label中的每一行
        for line in lines:  # 把lines中的数据逐行读取出来
            list = line.strip('\n').split(' ')
            newList = list[0] + ' ' + list[1] + ' ' + list[2] + ' ' + list[3] + ' ' + list[4] + '\n'
            fb.write(newList)

    # 旋转
    elif method == 'rotation':
        angle = random.uniform(-45, 45)
        print(angle)
        R = cv2.getRotationMatrix2D(angle=angle, center=(width / 2, height / 2), scale=1)
        # 变换图片
        img = cv2.warpAffine(img, R, dsize=(width, height), borderValue=(114, 114, 114))
        # 变换标签
        lines = label.readlines()  # 读出label中的每一行
        for line in lines:  # 把lines中的数据逐行读取出来
            list = line.strip('\n').split(' ')
            w, h = float(list[3]), float(list[4])
            x, y = float(list[1]) - w / 2, float(list[2]) - h / 2
            x1, y1, x2, y2, x3, y3, x4, y4 = x, y, x, y + h, x + w, y + h, x + w, y
            x1, y1 = rotate_xy(x1, y1, angle, 0.5, 0.5)
            x2, y2 = rotate_xy(x2, y2, angle, 0.5, 0.5)
            x3, y3 = rotate_xy(x3, y3, angle, 0.5, 0.5)
            x4, y4 = rotate_xy(x4, y4, angle, 0.5, 0.5)
            if angle < 0:
                nx, ny, nw, nh = x1, y4, x3 - x1, y2 - y4
            if angle >= 0:
                nx, ny, nw, nh = x2, y1, x4 - x2, y3 - y1
            nxc, nyc = nx + nw/2, ny + nh/2
            nxc, nyc, nw, nh = roundxywh(nxc, nyc, nw, nh)
            if boundary(nxc, nyc, nw, nh):
                newList = list[0] + ' ' + str(nxc) + ' ' + str(nyc) + ' ' + str(nw) + ' ' + str(nh) + '\n'
                fb.write(newList)

    # 缩放
    elif method == 'scale':
        # 对输入进来的图片进行缩放
        aw, ah = np.random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)
        nw, nh = int(width * aw), int(height * ah)
        img = cv2.resize(img, dsize=(nw, nh))

        # 变换图片
        #  img = cv2.resize(img, dsize=(nw, nh))
        # 变换标签
        lines = label.readlines()  # 读出label中的每一行
        for line in lines:  # 把lines中的数据逐行读取出来
            list = line.strip('\n').split(' ')
            nxc, nyc, nw, nh = float(list[1]), float(list[2]), float(list[3]), float(list[4])
            nxc, nyc, nw, nh = roundxywh(nxc, nyc, nw, nh)
            newList = list[0] + ' ' + str(nxc) + ' ' +str(nyc) + ' ' + str(nw) + ' ' + str(nh) + '\n'
            fb.write(newList)

    # 剪切
    elif method == 'shear':
        S = np.eye(3)
        sh = 40.0
        angle1 = random.uniform(-sh, sh)
        angle2 = random.uniform(-sh, sh)
        S[0, 1] = math.tan(angle1 * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(angle2 * math.pi / 180)  # y shear (deg)
        # 变换图片
        img = cv2.warpAffine(img, S[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # 平移
    elif method == 'translation':
        T = np.eye(3)
        num1 = random.uniform(-0.35, 0.35)
        num2 = random.uniform(-0.35, 0.35)
        T[0, 2] = num1 * width  # x translation (pixels)
        T[1, 2] = num2 * height  # y translation (pixels)
        # 变换图片
        img = cv2.warpAffine(img, T[:2], dsize=(width, height), borderValue=(114, 114, 114))
        # 变换标签
        lines = label.readlines()  # 读出label中的每一行
        for line in lines:  # 把lines中的数据逐行读取出来
            list = line.strip('\n').split(' ')
            w, h = float(list[3]), float(list[4])
            xc, yc = float(list[1]), float(list[2])
            nxc, nyc = xc + num1, yc + num2
            nxc, nyc, nw, nh = roundxywh(nxc, nyc, w, h)
            if boundary(nxc, nyc, w, h):
                newList = list[0] + ' ' + str(nxc) + ' ' + str(nyc) + ' ' + str(nw) + ' ' + str(nh) + '\n'
                fb.write(newList)

    # 透视变换
    elif method == 'perspective':
        P = np.eye(3)
        pe = 0.001
        P[2, 0] = random.uniform(-pe, pe)  # x perspective (about y)
        P[2, 1] = random.uniform(-pe, pe)  # y perspective (about x)
        img = cv2.warpPerspective(img, P, dsize=(width, height), borderValue=(114, 114, 114))

    fb.close()
    return img


def main(img_path, img_name, label_path, label_name, outpath):
    img = img_name[0:-4]  # 去除文件后缀名
    # ori_img = cv2.imread(filename=img_path)

    # 将原images与labels拷贝到要输出的文件夹中
    shutil.copy(img_path, outpath + '/images/' + img_name)  # 存原图
    shutil.copy(label_path, outpath + '/labels/' + label_name)  # 存原label


    # 进行数据增强   上下翻转、左右翻转、HSV增强、旋转、缩放、剪切、平移、透视、cutmix、cutout
    methodList = ['fliplr', 'flipud', 'hsv', 'scale', 'translation']
    for met in methodList:
        newImage_path = outpath + '/images/' + img + '_' + met + '.jpg'
        newLabel_path = outpath + '/labels/' + img + '_' + met + '.txt'

        out_img = data_augment(img_path, label_path, newLabel_path, method=met)

        cv2.imwrite(newImage_path, out_img)


def run(inpath, outpath):
    # inpath = '../yolov5-master/datasets/DataAugment/train/'
    #outpath = '../yolov5-master/datasets/DataAugment/train_SP/'
    #outpath = 'output/1/'
    image_dir = inpath+'images/'  # 原始图片路径
    label_dir = inpath+'labels/'  # 原始标签路径
    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)

    i = 0
    for img_name in images:
        i += 1
        img_path = image_dir + img_name
        for label_name in labels:
            if label_name[0:-4] == img_name[0:-4]:
                label_path = label_dir + label_name
                #os.remove(label_path)
                main(img_path, img_name, label_path, label_name, outpath)
                print('process:'+str(i) + '\n')


'''if __name__ == '__main__':
    run()'''