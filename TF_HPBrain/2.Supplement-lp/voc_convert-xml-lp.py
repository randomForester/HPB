'''
Annotations:
Annotations/LicensePlate00000001.xml
Annotations/LicensePlate00000002.xml
.
.

Images:
JPEGImages/LicensePlate00000001.jpg
JPEGImages/LicensePlate00000002.jpg
.
.
'''

import xml.etree.ElementTree as ET
from os import getcwd

classes = ["license plate", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
"O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def convert_annotation(image_id, list_file):
    in_file = open('Annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()
print(wd)

'''
image_ids = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015',
	     '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030']
'''

image_ids = open('numbers-lp.txt').read().strip().split()
list_file = open('AJ-lp.txt', 'w')

for image_id in image_ids:
    list_file.write('%s/JPEGImages/%s.jpg'%(wd, image_id))
    convert_annotation(image_id, list_file)
    list_file.write('\n')
list_file.close()

