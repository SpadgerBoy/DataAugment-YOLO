
import SinglePic as SP
import Mosaic
import DrowBox

if __name__ == '__main__':

    # '../yolov5-master/datasets/DataAugment/train/'存放的是原图

    inpath = '../yolov5-master/datasets/DataAugment/train/'
    outpath = '../yolov5-master/datasets/DataAugment/train_SP/'
    SP.run(inpath, outpath) # 生成新images+labels
    # DrowBox.run(outpath)  # 根据新的labels对新的images画出标记框，验证生成的labels是否有错


    inpath = '../yolov5-master/datasets/DataAugment/train/'
    outpath = '../yolov5-master/datasets/DataAugment/train_Mosaic/'
    Mosaic.run(inpath, outpath)
    # DrowBox.run(outpath)

