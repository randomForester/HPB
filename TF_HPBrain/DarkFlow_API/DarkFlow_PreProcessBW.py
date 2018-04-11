import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
from time import time as timer

import matplotlib.pyplot as plt

##########################
# Run DarkFlow           #
##########################

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.25,
    'gpu': 0.0
}

tfnet = TFNet(option)

##########################
# Processing Images (1)  #
##########################
# read the color image and covert to RGB
#img = cv2.imread('eagle.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('dog.jpg', cv2.IMREAD_COLOR)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# B & W image (using red channel only)
img = np.stack((img[:,:,0],img[:,:,0],img[:,:,0]),axis=2)

# use YOLO to predict the image
result = tfnet.return_predict(img)

result

img.shape

plt.imshow(img)
plt.show()

# pull out some info from the results

tl = (result[0]['topleft']['x'], result[0]['topleft']['y'])
print(tl)
br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])
print(br)
label = result[0]['label']

# add the box and label and display it
img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
plt.imshow(img)
plt.show()

#######################
# Pre-process (a)     #
#######################
#img = cv2.imread('eagle.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('dog.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# B & W image (using red channel only)
img = np.stack((img[:,:,0],img[:,:,0],img[:,:,0]),axis=2)

start = timer()

preprocessed = tfnet.framework.preprocess(img)
feed_dict = {tfnet.inp: [preprocessed]}
net_out = tfnet.sess.run(tfnet.out,feed_dict)[0]

processed = tfnet.framework.postprocess(net_out, img, False)

cv2.imwrite('out_dog.jpg', processed)
print('Elapsed time = ' + str(timer() - start) + ' s')

'''
fig = plt.figure(1)
fig.gca().imshow(processed)
plt.show()
'''

plt.imshow(processed)
plt.show()

#######################
# Pre-process (b)     #
#######################
#img = cv2.imread('eagle.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('dog.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# B & W image (using red channel only)
img = np.stack((img[:,:,0],img[:,:,0],img[:,:,0]),axis=2)

start = timer()

buffer_inp = list()
buffer_pre = list()

preprocessed = tfnet.framework.preprocess(img)

buffer_inp.append(img)
buffer_pre.append(preprocessed)

feed_dict = {tfnet.inp: buffer_pre}
net_out = tfnet.sess.run(tfnet.out,feed_dict)[0]

processed = tfnet.framework.postprocess(net_out, img, False)

cv2.imwrite('out_dog.jpg', processed)
print('Elapsed time = ' + str(timer() - start) + ' s')

plt.imshow(processed)
plt.show()

#######################
# Pre-process (c)     #
#######################
#img = cv2.imread('eagle.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('dog.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# B & W image (using red channel only)
img = np.stack((img[:,:,0],img[:,:,0],img[:,:,0]),axis=2)

start = timer()

h, w, _ = img.shape

'''
img = tfnet.framework.resize_input(img)
this_inp = np.expand_dims(img, 0)
feed_dict = {tfnet.inp : this_inp}
'''

preprocessed = tfnet.framework.preprocess(img)
feed_dict = {tfnet.inp: [preprocessed]}

net_out = tfnet.sess.run(tfnet.out, feed_dict)[0]
processed = tfnet.framework.postprocess(net_out, img, False)

###############
# Boxes Info  #
###############
out = tfnet.sess.run(tfnet.out, feed_dict)[0]
boxes = tfnet.framework.findboxes(out)

threshold = tfnet.FLAGS.threshold

boxesInfo = list()
#
for box in boxes:
    tmpBox = tfnet.framework.process_box(box, h, w, threshold)
    if tmpBox is None:
        continue
    boxesInfo.append({
        "label": tmpBox[4],
        "confidence": tmpBox[6],
        "topleft": {
            "x": tmpBox[0],
            "y": tmpBox[2]},
        "bottomright": {
            "x": tmpBox[1],
            "y": tmpBox[3]}
    })
#
boxesInfo
#

cv2.imwrite('out_dog.jpg', processed)
print('Elapsed time = ' + str(timer() - start) + ' s')

plt.imshow(processed)
plt.show()

############################
# Processing Images (2)    #
############################
colors = [tuple(255 * np.random.rand(3)) for i in range(7)]  #range(100)

for color in colors:
    print(color)

# read the color image and covert to RGB

stime = time.time()
img = cv2.imread('dog.jpg', cv2.IMREAD_COLOR)
#img = cv2.imread('eagle.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# B & W image (using red channel only)
img = np.stack((img[:,:,0],img[:,:,0],img[:,:,0]),axis=2)

results = tfnet.return_predict(img)

for color, result in zip(colors, results):
    tl = (result['topleft']['x'], result['topleft']['y'])
    br = (result['bottomright']['x'], result['bottomright']['y'])
    label = result['label']
    img = cv2.rectangle(img, tl, br, color, 7)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    print(result)

cv2.imshow('img', img)
print('FPS {:.1f}'.format(1 / (time.time() - stime)))  # Frame per seconds

plt.imshow(img)
plt.show()
