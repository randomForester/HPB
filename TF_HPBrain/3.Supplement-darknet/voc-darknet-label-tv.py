import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

#sets=[('2007', 'train')]
#sets=['train']

classes = ["license plate", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
#classes = ["license plate"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('annotation/%s.xml'%(image_id))
    out_file = open('labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

image_ids_train = open('ImageSets/Main/train.txt').read().strip().split()
image_ids_val   = open('ImageSets/Main/val.txt').read().strip().split()

list_file_train = open('lp_train.txt', 'w')
list_file_val   = open('lp_val.txt', 'w')

for image_id in image_ids_train:
    #list_file_train.write('/home/cesare/Github/cars_markus-all-train+val/image/%s.jpg\n'%(image_id))
    list_file_train.write('%s/image/%s.jpg\n'%(wd, image_id))
    convert_annotation(image_id)
list_file_train.close()

for image_id in image_ids_val:
    list_file_val.write('%s/image/%s.jpg\n'%(wd, image_id))
    convert_annotation(image_id)
list_file_val.close()
