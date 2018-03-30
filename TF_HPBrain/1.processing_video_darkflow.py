#####################################################
# Implementation of Video Processing using darkflow #
#####################################################
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
#####################################
# Options: Models, Weights, GPU etc #
#####################################
option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.25,
    'gpu': 0.0
}
#########################
# Processing Tensorflow #
#########################
tfnet = TFNet(option)
###############
# Input Video #
###############
capture = cv2.VideoCapture('../VideosHPB/IMAG0011.mp4')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]  #range(100)
'''
for color in colors:
    print(color)
'''
#############################
# Create VideoWriter object #
#############################
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#out = cv2.VideoWriter('output.avi',fourcc, 0.7, (640,480))
#################################################################
# These 2 lines can be removed if you don't have a 1080p camera #
#################################################################
'''
capture.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
'''
#################
# Run the Video #
#################
while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    if ret:
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))  # Frame per seconds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
