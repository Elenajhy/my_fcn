# coding:utf-8
import numpy as np
from PIL import Image
import matplotlib  # 增加画图需要import的包
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caffe
import cv2
import scipy.io as scio


# 1.load image,:switch to BGR, subtract mean, and make dims C x H x W for Caffe
def load_image(test_img_dir):
    im = Image.open(test_img_dir)  # 测试图片
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]
    in_ -= np.array((75.5052908510844, 90.55851244227091, 80.09322379964193),dtype=np.float32)  # 减均值(104.00698793,116.66876762,122.67891434) 
    in_ = in_.transpose((2, 0, 1))
    print "in.shape:", in_.shape  # (3,512,512)
    return in_


# 2.load and run net,take argmax for prediction
def load_net(in_, caffemodel_dir):
    net = caffe.Net('deploy.prototxt', caffemodel_dir,caffe.TEST)  # 网络描述文件和模型路径 caffemodel:./models/dstl_iter_100000.caffemodel
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)  # 第一维：N，batch size=1;*in_.shape:*表示接受任意多个参数放在一个元组中[1,(3,512,512)]
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score_sem'].data[0].argmax(axis=0)  # 修改score为score_sem
    return out


# 获取测试图片类别
def get_out_Class(out):
    print "test img size:", out.shape  # (512,512)
    labels = []
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            labels.append(out[i, j])
    labelSet = []
    for label in labels:
        labelSet.append(label)
    labelSet = list(set(labelSet))  # 类别集合
    print "label sum:", len(labelSet), "\nlabelSet:\n", labelSet


# 获取语义类mat类别
def getClass(file):
    labels = []
    mat = scio.loadmat(file)
    mat_S = mat.get('S')
    print "groundtruth(mat) size:", mat_S.shape  # (512L,512L)代表图片像素点
    for i in range(mat_S.shape[0]):  # i代表横轴x上的像素
        for j in range(mat_S.shape[1]):  # j代表纵轴y上的像素
            labels.append(mat_S[i, j])  # img[i,j]:输出一个数字代表的是灰度的类也就是语义类
    labelSet = []  # 灰度类别集合，去除重复的
    for label in labels:
        labelSet.append(label)
    labelSet = list(set(labelSet))
    sum = labelSet.__len__()
    print "labelSum:", sum
    print "label:", labelSet


# 获取图像分割准确率
def getSegRate(groudtruth_dir, out):
    mat_groundtruth = scio.loadmat(groundtruth_dir)
    mat_S = mat_groundtruth.get('S')
    segNum = 0
    gtNum = out.shape[0] * out.shape[1] # 512*512=262144
    for i in range(out.shape[0] ):
        for j in range(out.shape[1]):
            if out[i, j] == mat_S[i, j] :#忽略背景类像素点的判断
                segNum += 1
   # print "SegNum:", segNum
    segrate = 1 - (float)(abs(gtNum - segNum)) / gtNum  # 分割准确率
    print "Segment Rate:", segrate
    return segrate

# 保存图片
def save_image(out, out_dir):
    # cv2.imwrite("1_32_gray.tif",out)
    # pillow保存图片默认为彩色，变为灰色则：plt.imshow(out,cmap='gray')
    # plt.imshow(out)        #将生成的图片保存在当前目录下，为test_result.png
    # plt.axis('off')
    # plt.savefig('./15555_32.tif')     #将生成的图片保存在当前目录
    # 设置调色盘
    arr = out.astype(np.uint8)
    im = Image.fromarray(arr)
    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 11] = np.array(
        [[0,0,0],[255, 215, 0], [0, 255, 0], [192, 192, 192], [192, 0, 0], [128, 128, 128], [255, 160, 0], [200, 165, 0],
         [0,100, 0], [0, 150, 150], [0, 192, 255]], dtype='uint8').flatten()  # 每个类定义一个RGB
    im.putpalette(palette)
    im.show()
    im.save(out_dir)  # '5_32_palette.tif'

if __name__ == '__main__':
    test_img_dir = "../data/dstl/Images/" + "1" + ".tif"
    in_ = load_image(test_img_dir)  # 1.将测试图片加载
    caffemodel_dir = "./models/dstl_iter_150000.caffemodel"
    out = load_net(in_, caffemodel_dir)  # 2.运行模型并预测获得预测图片矩阵
    groundtruth_dir = "../data/dstl/graySemLabels/" + "1" + ".mat"  # 获取对应的groundtruth标注的mat文件
    getClass(groundtruth_dir)  # 得到groundtruth类别
    get_out_Class(out)  # 得到预测类别
    getSegRate(groundtruth_dir, out)  # 获取精度
    out_dir = "1" + "_16_palette.tif"
    save_image(out, out_dir)  # 将预测矩阵保存为图片
    '''
    # 获取每个模型在测试集上的平均分割精度
   # for i in range(2,9,2):
    i=15
    caffemodel_dir = "./models/dstl_iter_"+str(i)+"0000.caffemodel"  # 2/4/6/8/10 0000.caffemodel
    SegRate_sum = 0
    result_txt=open("./rate/result"+str(i)+".txt","w")  #获得的结果分别写入result2/4/6/8/10 .txt
    result_txt.truncate()
    result_txt.write("{}           {}\n".format("image","segment rate"))
    with open("../data/dstl/test.txt", "r") as f:
        linelist = f.readlines()
        sum = len(linelist)
        for line in linelist:
            image_num = line.strip("\n")
            current_img=image_num+".tif"
            in_ = load_image("../data/dstl/Images/" + current_img)
            out = load_net(in_, caffemodel_dir)
            groundtruth_dir = "../data/dstl/graySemLabels/" + image_num + ".mat"
            segrate = getSegRate(groundtruth_dir, out)
            print current_img ," : ", segrate  # 6.tif : 0.65..
            result_txt.write("{}        {}\n".format(str(image_num) + ".tif", str(segrate)))
            SegRate_sum += segrate
        average_rate = SegRate_sum / sum
        result_txt.write("sum : {}\n average seg rate: : {}\n".format(sum, str(average_rate)))
        result_txt.close()
        print "compute end !"
     '''
