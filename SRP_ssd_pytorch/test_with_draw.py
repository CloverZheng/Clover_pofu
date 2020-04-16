from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import time

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_VOC_6000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    
    #num_images = len(testset)
    num_images=5
    for i in range(num_images):
        test_s=time.time()
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        filename = save_folder+'test'+img_id+'.xml'
        with open(filename, mode='w') as f:
            f.write('<?xml version="1.0" ?>\n'+
			'<annotation>\n'+
            '<folder>KIITI</folder>\n'+
			'<filename>'+img_id+'.png</filename>\n'+
			'<object>\n'
			)
            #for box in annotation:
            #    f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            score=detections[0, i, j, 0]
            if detections[0, i, j, 0] >= 0.1:
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    #f.write(str(pred_num)+' label: '+label_name+' score: ' +
                    #        str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                    f.write('<name>'+label_name+'</name>\n'+
					        '<bndbox>\n'+
							'<xmin>'+str(int(pt[0]))+'</xmin>\n'+
							'<ymin>'+str(int(pt[1]))+'</ymin>\n'+
							'<xmax>'+str(int(pt[2]))+'</xmax>\n'+
							'<ymax>'+str(int(pt[3]))+'</ymax>\n'+
							'<score>'+str(score)+'</score>\n'+
							'</bndbox>\n')
                j += 1
        with open(filename, mode='a') as f:
            f.write('</object>\n</annotation>')		
        test_e=time.time()
        print('test time:',test_e-test_s,' sec')

def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    t_build=time.time()
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    #print('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    t_buile_ed=time.time()
    print('loading net...',t_buile_ed-t_build,'sec')
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
