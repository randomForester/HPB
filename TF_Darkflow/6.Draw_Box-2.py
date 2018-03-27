import sys
print("Python version: {}".format(sys.version))

import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector
#from generate_xml import write_xml

# global constants
img = None
tl_list = []
br_list = []
object_list = []

# constants
image_folder = 'images'
savedir = 'annotations'
obj = 'fidget_spinner'

def line_select_callback(clk):
    print(clk.xdata, clk.ydata)

if __name__ == '__main__':
    for n, image_file in enumerate(os.scandir(image_folder)):
        img = image_file
        fig, ax = plt.subplots(1)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(250, 120, 1280, 1024)
        image = cv2.imread(image_file.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        plt.connect('button_press_event',line_select_callback)
        plt.show()
