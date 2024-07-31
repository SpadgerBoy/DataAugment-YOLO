'''
Mosaic数据增强
'''

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


# 将xyxy转换为yolov5并归一化
def xyxy2xywh(x1, y1, x2, y2, height, width):
    w = round((x2 - x1) / width, 6)
    h = round((y2 - y1) / height, 6)
    xc = round((x1 + x2) / 2 / width, 6)
    yc = round((y1 + y2) / 2 / height, 6)
    return xc, yc, w, h


def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox


def get_random_data(b_data, input_shape, hue=.1, sat=1.5, val=1.5):
    h, w = input_shape
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)  # 0.6
    scale_high = scale_low + 0.2  # 0.8

    image_datas = []
    box_datas = []
    index = 0

    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]  # [0, 0, 243, 243]
    place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]  # [0, 216, 243, 0]
    print("place:", place_x, place_y)

    for i in range(4):
        idx = i
        img, box, img_path, img_name = b_data[i]
        # print(img_path, boxes)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img, mode="RGB")

        # 图片的大小
        iw, ih = image.size

        # 是否翻转图片
        flip = rand() < .5
        if flip and len(box) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0, 2]] = iw - box[:, [2, 0]]

        # 对输入进来的图片进行缩放
        new_ar = w / h
        scale = (scale_low + scale_high) / 2
        # scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 进行色域变换
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image = hsv_to_rgb(x)

        image = Image.fromarray((image * 255).astype(np.uint8))
        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255

        index = index + 1
        box_data = []
        # 对box进行重新处理，处理越界问题。
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        image_datas.append(image_data)
        box_datas.append(box_data)

        img = Image.fromarray((image_data * 255).astype(np.uint8))
        for j in range(len(box_data)):
            thickness = 3
            left, top, right, bottom = box_data[j][0:4]
            draw = ImageDraw.Draw(img)
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
        # img.show()

        # img.save("outputs/3/images/" + img_name[0:-4] + '_' + str(idx + 1) + '.jpg')

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)
    return new_image, new_boxes


def get_4_data(images4, image_dir, label_dir):
    batch_data = []
    for img_name in images4:
        img_path = image_dir + img_name
        # print(img_path)
        label_path = label_dir + img_name[:-4] + ".txt"

        img = cv2.imread(img_path)

        # 读label
        gt_boxes = []
        lines = []
        with open(label_path) as fp:
            for item in fp.readlines():
                lines.append(item.strip().split())
        lines = [v for v in lines if v]

        img_h, img_w = img.shape[:2]
        for item in lines:
            item = [float(v) for v in item]
            [cls, cx, cy, bw, bh] = item
            x1 = max(0, int((cx - bw / 2) * img_w))
            y1 = max(0, int((cy - bh / 2) * img_h))
            x2 = min(int((cx + bw / 2) * img_w), img_w - 1)
            y2 = min(int((cy + bh / 2) * img_h), img_h - 1)
            gt_boxes.append([x1, y1, x2, y2, int(cls)])

        batch_data.append([img, np.array(gt_boxes), img_path, img_name])
    return batch_data


def run(inpath,outpath):
    #inpath = '../yolov5-master/datasets/DataAugment/train/'
    #outpath = '../yolov5-master/datasets/DataAugment/train_mosaic/'  # 输出路径
    #outpath = 'output/3/'
    image_dir = inpath+'images/'  # 原始图片路径
    label_dir = inpath+'labels/'  # 原始标签路径
    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)

    imgs_path = []
    for img_name in images:
        img_path = image_dir + img_name
        if img_name.endswith(".jpg") and os.path.exists(label_dir + img_name[:-4] + ".txt"):
            imgs_path.append(img_path)
    print("label img cnt:", len(imgs_path), imgs_path)

    if len(imgs_path) < 4:
        print("数据不足！")
        return

    images4 = []
    a = 0
    for num in range(len(images) * 4 + 1):  # a是image在images中的位置,num/4是合成图片的总量

        image = images[a]

        # 每次合成使用4张图片
        if len(images4) == 4:
            batch_data = get_4_data(images4, image_dir, label_dir)

            # image_data, box_data = get_random_data(batch_data, [765, 1360])
            image_data, box_data = get_random_data(batch_data, [608, 800])
            new_img = Image.fromarray((image_data * 255).astype(np.uint8))

            # 生成的图片序号与名称
            file_num = str(int(num/4)).zfill(5)
            file_name = 'mosaic_' + file_num
            print(file_name+'.jpg')
            newImg_path = outpath + 'images/' + file_name + '.jpg'
            newLabel_path = outpath + 'labels/' + file_name + '.txt'

            if os.path.exists(newLabel_path):
                os.remove(newLabel_path)

            fb = open(newLabel_path, mode='a', encoding='utf-8')  # 创建一个新的txt的文本用于存储新的标签
            for j in range(len(box_data)):
                thickness = 3
                # print(box_data[j])
                left, top, right, bottom, cls = box_data[j][0:5]
                cls = int(cls)
                xc, yc, w, h = xyxy2xywh(left, top, right, bottom, 608, 800)
                labelline = str(cls) + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(w) + ' ' + str(h) + '\n'
                fb.write(labelline)
                draw = ImageDraw.Draw(new_img)
                # for i in range(thickness):
                #    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
            new_img.save(newImg_path)
            # print('num = ' + str(num))
            images4 = []  # 将列表清空，存储下4个图片
        images4.append(image)
        # print('a = ' + str(a))

        a += 1
        if a == len(images):
            a = 0


'''if __name__ == "__main__":
    run()'''
