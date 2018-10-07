import os
import glob
os.system
import PIL
import PIL.Image as Image

d = 0
test_Path = r'/home/nechk/Downloads/cars-test/cars-test'
with open((test_Path + '.txt'),'r') as fobj:

    for line in fobj:
        image_List = [[num for num in line.split()] for line in fobj]

        for images in image_List:
            commands = ['./darknet detector test cfg/voc-lp-11c.data cfg/yolov3-voc-lp-11c.cfg backup/yolov3-voc-11-48-randomzero_final.weights', images[0]]
            os.system(' '.join(commands))
            predicted_image = Image.open("/home/nechk/NECHK-Results/darknet-AB/predictions.jpg")

            output = "/home/nechk/Downloads/cars-output/predicted_image%d.jpg"%d
            predicted_image.save(output)
            d+=1
