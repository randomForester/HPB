#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np
from timeit import default_timer as timer

def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes
    ###############################
    input_path[-4:] == '.mp4' # do detection on a video
    video_out = output_path + input_path.split('/')[-1]
    video_reader = cv2.VideoCapture(input_path)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'XVID'),
                               50.0,
                               (frame_w, frame_h))
    # the main loop
    batch_size  = 1
    images      = []
    start_point = 0 #%
    show_window = True
    #
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    #
    cnt = 0
    #
    while True:
    #for i in tqdm(range(nb_frames)):
        return_value, image = video_reader.read()

        if return_value == True:

            cnt += 1
            #with open('outputFrames.csv', 'a') as f:
            #    print('{}, {}'.format(cnt - 1, nb_frames), file=f)
            #print('Frame Number (Frame)   {}, {}'.format(cnt - 1, nb_frames))
            print('{}, {}'.format(cnt - 1, nb_frames))

        #if (float(i+1)/nb_frames) > start_point/100.:
            images += [image]

            #if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
            # predict the bounding boxes
            batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

            for i in range(len(images)):
                # draw bounding boxes on the image using labels
                draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)
                #
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0
                cv2.putText(images[i], text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
                #
                # show the video with detection bounding boxes
                if show_window: cv2.imshow('video with bboxes', images[i])

                # write result to the output video
                video_writer.write(images[i])
            images = []
        if show_window and cv2.waitKey(1) == 27: break  # esc to quit

    if show_window: cv2.destroyAllWindows()
    video_reader.release()
    video_writer.release()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')

    args = argparser.parse_args()
    _main_(args)
