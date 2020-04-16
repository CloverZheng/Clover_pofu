import os
import xml.dom.minidom
import cv2 as cv
import argparse
from data import VOC_ROOT, VOC_CLASSES as labelmap
from data import VOCAnnotationTransform, VOCDetection
ImgPath = '/home/guoxiong/KITTI_xml/VOC2007/JPEGImages/'
AnnoPath = '/home/guoxiong/clover/ssd.pytorch-master/eval/'  #xml文件地址
#AnnoPath = '/home/guoxiong/KITTI_xml/VOC2007/Annotations/'
save_path = './myimg'

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
args = parser.parse_args()



def draw_anchor(ImgPath,AnnoPath,save_path):
    # load data
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())
    imagelist = os.listdir(ImgPath)
    cnt=5
    #for image in imagelist:
    for i in range(cnt):
        image,annotation=testset.pull_anno(i)
        #image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + image+'.png'
        xmlfile = AnnoPath + 'test'+image + '.xml'
        #xmlfile = AnnoPath + image + '.xml'
        #xmlfile = AnnoPath +image + '.xml'
        # print(image)
        # 打开xml文档
        DOMTree = xml.dom.minidom.parse(xmlfile)
        # 得到文档元素对象
        collection = DOMTree.documentElement
        # 读取图片
        img = cv.imread(imgfile)
 
        filenamelist = collection.getElementsByTagName("filename")
        filename = filenamelist[0].childNodes[0].data
        print(filename)
        # 得到标签名为object的信息
        objectlist = collection.getElementsByTagName("object")
 
        for objects in objectlist:
            # 每个object中得到子标签名为name的信息
            namelist = objects.getElementsByTagName('name')           
            name_idx=0
            bndbox = objects.getElementsByTagName('bndbox')
            # print(bndbox)
            for box in bndbox:
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')   #注意坐标，看是否需要转换
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), thickness=2)
                # 通过此语句得到具体的某个name的值
                objectname = namelist[name_idx].childNodes[0].data
                cv.putText(img, objectname, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0,255),
                           thickness=1)
                name_idx+=1
                #cv.imshow(filename, img)#这个要安装Xmanager才可以看
                cv.imwrite(save_path+'/'+filename, img)   #save picture
				
draw_anchor(ImgPath,AnnoPath,save_path)