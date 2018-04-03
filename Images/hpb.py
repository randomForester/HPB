#################
# Python basics #
#################
x = 3
print(x)

def compute_number_of_days(age):
    # this function roughly computes the number of days a person has lived
    days = age * 365
    return days

days = compute_number_of_days(22)
print(days)

def print_congratulations(age, name):
    days = compute_number_of_days(age)
    print('Hello ' + name + '! You have already lived on this planet for ' + str(days) + ' days!')

print_congratulations(41, 'Cesare')

ages_parents = [51, 52]
ages_children = [2, 4, 10]

ages_family = ages_parents + ages_children
print(ages_family)

total_age = 0
for age in ages_family:
    total_age = total_age + age
print(total_age)

ages_family = [51, 52, 2, 4, 10]
names = ['Patrick', 'Maria', 'Emma', 'Jordi', 'Vasiliy']
print(len(names))

for i in range(0, len(names)):
    current_name = names[i]
    current_age = ages_family[i]
    print_congratulations(current_age, current_name)

for i in range(3, 7):
    for j in range (100, 103):
        mul = i * j
        print('If you multiply ' + str(i) + ' by ' + str(j) + ', you get ' + str(mul))
###################
# Python Matrices #
###################
import numpy as np

x = np.random.rand()
print(x)
A = np.matrix([[1, 2], [3, 4], [5, 6]])
print(A)

print(np.shape(A))
b = np.array([1, 2])

r = A.dot(b)
print(r)
print(r.T)
print(np.max(r))
print(np.argmax(r))

#######################
# Working with images #
#######################
import numpy as np
# This is another way of loading a package. You can use it when you don't need
# to load the whole package, but only some parts of it. We need misc from
# reading an image file
from scipy import misc
# This is a plotting tool for python. We will use it for image visualization
import matplotlib.pyplot as plt
# This is a special command for Jupyter notebook. It forces plots to be refreshed
# when you recompute a cell
#%matplotlib inline
#
clown = misc.imread('/home/cesare/Github/04_python/image/clown.jpg')
clown = clown / 255.0
plt.imshow(clown)
#
red = np.zeros(np.shape(clown))
# Setting red channel (that has index 0) to 1
red[:,:,0] = 1
plt.imshow(red)
#
r = clown - red
# The np.linalg.norm function computes the norm of a vector. We are giving it a tensor of size (200, 185, 3).
# By default, it will give one number that will be the norm of all items in this tensor. However, if we provide
# the axis argument, the function will only compute norm in the given dimension. If we set axis to 2, we get a
# matrix of size (200, 185), every element of which is a norm of (r, g, b) values of a corresponding pixel.
d = np.linalg.norm(r, axis=2)
# When given a matrix with values, the imshow function color-maps them.
plt.imshow(d)
#
maxind = np.argmin(d)
# We obtain the height and width of the image
clown_hw = np.shape(clown)[0:2]
(y, x) = np.unravel_index(maxind, clown_hw)
#
plt.scatter(x, y)
plt.imshow(clown)

############################################
# 1.Requantize an image to n gray values   #
############################################
#%matplotlib inline
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
#
def read_input_image(input_file_name):

    # Set the flag argument to 0 so that the output has only one channel.
    return cv2.imread(input_file_name, 0).astype('float')
#
def write_output_image_to_file(output_image, output_file_name):

    cv2.imwrite(output_file_name, output_image)
    return
#
def map_to_unit_range(input_image, input_image_min, input_image_max):

    # Replace me!
    return input_image
#
def interval_ID_uniform_quantizer(image_unit_range, n_intervals):

    # Fill in the missing code at this point.

    # Correct me!
    return image_unit_range.astype(int)
#
def representative_values_entire_output_range(n_intervals):

    if n_intervals == 1:
        # Corner case: we cannot cover a range with only one value, so we assign the mean of the output range.
        representative_values_table = np.array([round((0 + 255) / 2.0)], dtype=np.uint8)
    else:
        # Replace me!
        representative_values_table = np.zeros(n_intervals, dtype=np.uint8)

    return representative_values_table
#
# Full path of the input image file.
input_file = '/home/cesare/Github/04_python/image/stpeter.png'
# Read input image into a variable.
I_input = read_input_image(input_file)

# Number of output gray values. Choose any integer from 1 up to 256.
n = 2
#
# First, determine minimum and maximum value of the input.
input_min = np.min(I_input)
input_max = np.max(I_input)

# Print the minimum and maximum value of the input.
print('The minimum gray level intensity of the input image',\
    'is {:.0f} and the respective maximum intensity is {:.0f}.'.format(input_min, input_max))

# Next, we map the used input range [|input_min|, |input_max|] to the unit range [0, 1],
# so that subsequent requantization assumes a straightforward form.
I_unit_range = map_to_unit_range(I_input, input_min, input_max)
I_interval_IDs = interval_ID_uniform_quantizer(I_unit_range, n)
#
representative_values = representative_values_entire_output_range(n)

# Print the representative values for all |n| intervals in ascending order.
print('The representative gray values for the output image are:')
print(representative_values)
I_output = np.zeros_like(I_input, dtype=np.uint8)
#
# Leave this part as is.
plt.close('all')
plt.ion()
f, axes_array = plt.subplots(1, 2, figsize=(18, 16))
axes_array[0].set_title('Input image', fontsize=12)
axes_array[0].imshow(I_input, cmap=plt.cm.gray)
axes_array[0].tick_params(bottom='off', labelbottom='off', left='off', labelleft='off')
axes_array[1].set_title('Output requantized image, n={:d}'.format(n), fontsize=12)
axes_array[1].imshow(I_output, cmap=plt.cm.gray)
axes_array[1].tick_params(bottom='off', labelbottom='off', left='off', labelleft='off')
#
############################################
# 2.Spatial sampling                       #
############################################
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib

def read_input_image(input_file_name):
    """
    Read an image from disk
    """

    # Set the flag argument to 0 so that the output has only one channel.
    return cv2.imread(input_file_name, 0).astype('float')

def write_output_image_to_file(output_image, output_file_name):
    """
    Write an image to disk
    """

    cv2.imwrite(output_file_name, output_image)
    return

def check_size(input_image, factor):
    """
    Check if the subsampling factor is too large.
    Input parameters:
        input_image: the input image
        factor: the required sub-sampling factor
    """

    condition1 = ((input_image.shape[0] // factor) is 0)
    condition2 = ((input_image.shape[1] // factor) is 0)

    if condition1 or condition2:
        print('Error! Subsampling rate is too large.')
        return 0

    else:
        print('Sub-sampling factor is permissible.')
        return 1


def subsample_image(input_image, factor):
    """
    Subsample the input image with the requested subsampling factor.
    Input parameters:
        input_image: the input image
        factor: the required sub-sampling factor
    Output:
        the sub-sampled image
    """

    # ************************************
    # CORRECT ME!
    # ************************************
    # Currently, the output image is just being set to the input image.
    # Replace this with the appropriate sub-sampling code
    # Hint: You may do this using a double for loop, but there is also a way to do this with one line of code!
    # ************************************
    output_image = input_image

    return output_image

def gaussian_filter_image(input_image, sigma):
    """
    Apply a gaussian blurring to the image
    Input parameters:
        input_image: the input image
        sigma: strength of the required gaussian blurring
    Output:
        gaussian blurred image
    """

    # ************************************
    # CORRECT ME!
    # ************************************
    # Currently, the output image is just being set to the input image.
    # Replace this with the appropriate code for smooth the image
    # Hint: Look at the modules being imported in the first cell of the notebook.
    # ************************************
    output_image = input_image

    return output_image

def print_image_size(image,image_name):
    """
    Print the size of the image
    Input parameters:
        image: the image whose size is required
        image_name: string to indicate the image content
    """

    print('Size of {:s}: {:d}, {:d}.'.format(image_name, image.shape[0], image.shape[1]))


def display_image(image,title):
    """
    Display an image
    Input parameters:
        image: the image to be displayed
        title: title of the figure
    """

    fig = matplotlib.pyplot.gcf()
    DPI = fig.get_dpi()
    plt.figure(figsize=(image.shape[0]/float(DPI),image.shape[1]/float(DPI)))
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(title)
    plt.show()

def main_function(input_file, factor, sigma):
    """
    Input parameters:
        input_file: path to the input image
        factor: sub-sampling factor
        sigma: strength of the gaussian blurring
    """

    # Read the input image
    I_input = read_input_image(input_file)

    # display the input image
    display_image(I_input, 'Input image')

    # Print the size of the input image
    print_image_size(I_input, 'Input image')

    print('\n==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ====')

    # print the requested sub-sampling factor
    print('Requested sub-sampling factor: {:d}.'.format(factor))

    # check if the sub-sampling factor is permissible
    if check_size(I_input, factor) is 0:
        return -1

    print('\n==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ====')

    # if requested, blur the image using a gaussian filter
    if (sigma is not 0):
        print('Applying a gaussian blur to the image.')
        I_input = gaussian_filter_image(I_input, sigma)
        display_image(I_input, 'Gaussian blurred image')
    else:
        print('Using the original image, without gaussian blurring.')

    print('\n==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ====')

    # Sub-sample the image
    I_output = subsample_image(I_input, factor)

    # display the sub-sampled image
    display_image(I_output, 'Sub-sampled image')

    # Print the size of the subsampled image
    print_image_size(I_output, 'Sub-sampled image')

input_file = '/home/cesare/Github/04_python/image/lenna.png'
factor = 3
sigma = 0

# Call the main_function
main_function(input_file, factor, sigma)

##############
# Solutions  #
##############

############################################
# 1.Requantize an image to n gray values   #
############################################
#%matplotlib inline
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_input_image(input_file_name):

    # Set the flag argument to 0 so that the output has only one channel.
    return cv2.imread(input_file_name, 0).astype('float')

def write_output_image_to_file(output_image, output_file_name):

    cv2.imwrite(output_file_name, output_image)
    return

def map_to_unit_range(input_image, input_image_min, input_image_max):

    return (input_image - input_image_min) / (input_image_max - input_image_min)

def interval_ID_uniform_quantizer(image_unit_range, n_intervals):

    image_interval_IDs = np.floor(n_intervals * image_unit_range + 1)

    image_interval_IDs = np.minimum(image_interval_IDs, n_intervals)

    return image_interval_IDs.astype(int)

def representative_values_entire_output_range(n_intervals):

    if n_intervals == 1:
        # Corner case: we cannot cover a range with only one value, so we assign the mean of the output range.
        representative_values_table = np.array([round((0 + 255) / 2.0)], dtype=np.uint8)
    else:
        interval_IDs = np.arange(1, n_intervals + 1)
        representative_values_table = np.rint((interval_IDs - 1) * (255.0 / (n_intervals - 1))).astype(np.uint8)

    return representative_values_table

# Full path of the input image file.
input_file = '/home/cesare/Github/04_python/image/stpeter.png'
# Read input image into a variable.
I_input = read_input_image(input_file)

# Number of output gray values. Choose any integer from 1 up to 256.
n = 2

# First, determine minimum and maximum value of the input.
input_min = np.min(I_input)
input_max = np.max(I_input)

# Print the minimum and maximum value of the input.
print('The minimum gray level intensity of the input image',\
    'is {:.0f} and the respective maximum intensity is {:.0f}.'.format(input_min, input_max))


# Next, we map the used input range [|input_min|, |input_max|] to the unit range [0, 1],
# so that subsequent requantization assumes a straightforward form.
I_unit_range = map_to_unit_range(I_input, input_min, input_max)

I_interval_IDs = interval_ID_uniform_quantizer(I_unit_range, n)

representative_values = representative_values_entire_output_range(n)

# Print the representative values for all |n| intervals in ascending order.
print('The representative gray values for the output image are:')
print(representative_values)

I_output = representative_values[I_interval_IDs - 1]

plt.close('all')
plt.ion()
f, axes_array = plt.subplots(1, 2, figsize=(18, 16))
axes_array[0].set_title('Input image', fontsize=12)
axes_array[0].imshow(I_input, cmap=plt.cm.gray)
axes_array[0].tick_params(bottom='off', labelbottom='off', left='off', labelleft='off')
axes_array[1].set_title('Output requantized image, n={:d}'.format(n), fontsize=12)
axes_array[1].imshow(I_output, cmap=plt.cm.gray)
axes_array[1].tick_params(bottom='off', labelbottom='off', left='off', labelleft='off')
############################################
# 2.Spatial sampling (Sol.)                #
############################################
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib

def read_input_image(input_file_name):
    """
    Read an image from disk
    """

    # Set the flag argument to 0 so that the output has only one channel.
    return cv2.imread(input_file_name, 0).astype('float')

def write_output_image_to_file(output_image, output_file_name):
    """
    Write an image to disk
    """

    cv2.imwrite(output_file_name, output_image)
    return

def check_size(input_image, factor):
    """
    Check if the subsampling factor is too large.
    Input parameters:
        input_image: the input image
        factor: the required sub-sampling factor
    """

    condition1 = ((input_image.shape[0] // factor) is 0)
    condition2 = ((input_image.shape[1] // factor) is 0)

    if condition1 or condition2:
        print ('Error! Subsampling rate is too large.')
        return 0

    else:
        print('Sub-sampling factor is permissible.')
        return 1

def subsample_image(input_image, factor):
    """
    Subsample the input image with the requested subsampling factor.
    Input parameters:
        input_image: the input image
        factor: the required sub-sampling factor
    Output:
        the sub-sampled image
    """

    output_image = input_image[::factor, ::factor]

    return output_image

def gaussian_filter_image(input_image, sigma):
    """
    Apply a gaussian blurring to the image
    Input parameters:
        input_image: the input image
        sigma: strength of the required gaussian blurring
    Output:
        gaussian blurred image
    """

    output_image = gaussian_filter(input_image, sigma=sigma)

    return output_image

def print_image_size(image,image_name):
    """
    Print the size of the image
    Input parameters:
        image: the image whose size is required
        image_name: string to indicate the image content
    """

    print('Size of {:s}: {:d}, {:d}.'.format(image_name, image.shape[0], image.shape[1]))

def display_image(image,title):
    """
    Display an image
    Input parameters:
        image: the image to be displayed
        title: title of the figure
    """

    fig = matplotlib.pyplot.gcf()
    DPI = fig.get_dpi()
    plt.figure(figsize=(image.shape[0]/float(DPI),image.shape[1]/float(DPI)))
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(title)
    plt.show()

def main_function(input_file, factor, sigma):

    # Read the input image
    I_input = read_input_image(input_file)

    # display the input image
    display_image(I_input, 'Input image')

    # Print the size of the input image
    print_image_size(I_input, 'Input image')

    print('\n==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ====')

    # print the requested sub-sampling factor
    print('Requested sub-sampling factor: {:d}.'.format(factor))

    # check if the sub-sampling factor is permissible
    if check_size(I_input, factor) is 0:
        return -1

    print ('\n==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ====')

    # if requested, blur the image using a gaussian filter
    if (sigma is not 0):
        print ('Applying a gaussian blur to the image.')
        I_input = gaussian_filter_image(I_input, sigma)
        display_image(I_input, 'Gaussian blurred image')
    else:
        print('Using the original image, without gaussian blurring.')

    print ('\n==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ====')

    # Sub-sample the image
    I_output = subsample_image(I_input, factor)

    # display the sub-sampled image
    display_image(I_output, 'Sub-sampled image')

    # Print the size of the subsampled image
    print_image_size(I_output, 'Sub-sampled image')

# Try playing around with different values of the sub-sampling factor and the strength of the blurring
input_file = '/home/cesare/Github/04_python/image/lenna.png'
factor = 3
sigma = 1

# Call the main_function
main_function(input_file,factor,sigma)
############################################
# 3.Canny Edge Detector (Sol.)             #
############################################
import numpy as np
import scipy
import scipy.misc
from scipy.ndimage.filters import gaussian_filter,convolve
import matplotlib.pyplot as plt
from scipy import *
from scipy.ndimage import *
#%matplotlib inline
im=scipy.misc.imread('/home/cesare/Github/04_python/image/zurlim.png',mode='F')
print (im.shape)
plt.figure(figsize=(7,7))
plt.imshow(im,cmap='gray')
plt.show()
####### Gaussian Smooth Image #######
blurred_im = gaussian_filter(im, sigma=2,order=0,mode='reflect')
print (blurred_im.shape)

###### Gradients x and y (Sobel filters) ######
im_x = convolve(blurred_im,[[-1,0,1],[-2,0,2],[-1,0,1]])
im_y = convolve(blurred_im,[[1,2,1],[0,0,0],[-1,-2,-1]])

###### gradient and direction ########
gradient = np.power(np.power(im_x, 2.0) + np.power(im_y, 2.0), 0.5)
theta = np.arctan2(im_y, im_x)

###### Parameters #####
thresh=50;
thresholdEdges = (gradient > thresh)
plt.figure(figsize=(7,7))
plt.title('Edge image after Threshold criteria')
plt.imshow(thresholdEdges,cmap='gray')
plt.show()
###### Convert to degree ######
theta = 180 + (180/np.pi)*theta #

###### Quantize angles ######
x_0,y_0 = where(((theta<22.5)+(theta>157.5)*(theta<202.5)  +(theta>337.5)) == True)
x_45,y_45 = where(((theta>22.5)*(theta<67.5) +(theta>202.5)*(theta<247.5)) == True)
x_90,y_90 = where(((theta>67.5)*(theta<112.5) +(theta>247.5)*(theta<292.5)) == True)
x_135,y_135 = where(((theta>112.5)*(theta<157.5) +(theta>292.5)*(theta<337.5)) == True)

theta[x_0,y_0] = 0        # E-W
theta[x_45,y_45] = 1      # NE
theta[x_90,y_90] = 2      # N-S
theta[x_135,y_135] = 3    # NW

###### Non-maximum suppression ########
grad_supp = np.zeros((gradient.shape[0],gradient.shape[1]))
for r in range(im.shape[0]):
    for c in range(im.shape[1]):

        #Suppress pixels at the image edge
        if r == 0 or r == im.shape[0]-1 or c == 0 or c == im.shape[1] - 1:
            grad_supp[r, c] = 0
            continue

        ###### Thresholding #######
        if gradient[r, c]<thresh:
            grad_supp[r, c] = 0
            continue

        ######### NMS ##########
        tq = theta[r, c]
        if tq == 0: # E-W
            if gradient[r, c] >= gradient[r, c-1] and gradient[r, c] >= gradient[r, c+1]:
                grad_supp[r, c] = 1
        if tq == 1: # NE
            if gradient[r, c] >= gradient[r-1, c+1] and gradient[r, c] >= gradient[r+1, c-1]:
                grad_supp[r, c] = 1
        if tq == 2: # N-S (vertical)
            if gradient[r, c] >= gradient[r-1, c] and gradient[r, c] >= gradient[r+1, c]:
                grad_supp[r, c] = 1
        if tq == 3: # NW
            if gradient[r, c] >= gradient[r-1, c-1] and gradient[r, c] >= gradient[r+1, c+1]:
                grad_supp[r, c] = 1

strongEdges = (grad_supp > 0)

plt.figure(figsize=(7,7))
plt.title('Edge Image after Threshold and NMS')
plt.imshow(strongEdges,cmap='gray')
plt.show()
# Plotting of results
# No need to change it
f, ax_arr = plt.subplots(1, 2, figsize=(18, 16))
ax_arr[0].set_title("Input Image")
ax_arr[1].set_title("Canny Edge Detector")
ax_arr[0].imshow(im, cmap='gray')
ax_arr[1].imshow(strongEdges, cmap='gray')
