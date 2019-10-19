import matplotlib.pyplot as plt
from skimage import data,color, morphology,feature,segmentation,measure
import  cv2
import numpy as np
#
# img = plt.imread('../image/SK_means/lung/01.png')
# img = plt.imread('../image/SK_means/eyes/04.png')
# img = plt.imread('../image/SK_means/brain/1.png')
# img = plt.imread('../image/SK_means/liver/01.png')
#
# img = color.rgb2gray(img)
#
# img = (img > 0.5) * 1
#
# edgs = feature.canny(img,sigma=2,low_threshold=50,high_threshold=100)
#
# chull = morphology.convex_hull_image(img)
#
#
#
# img = segmentation.clear_border(img)
# label = measure.label(img,connectivity=None)
# img=morphology.remove_small_objects(label,min_size=119000,connectivity=3)
# img = (img > 0.5) * 1



def boundary_processing(path,min_size=500,conversion=False):
    img = cv2.imread(path)
    img = color.rgb2gray(img)
    if conversion == False:
        img = (img > 0.5) * 1
    else:
        img = (img < 0.5) * 1
    img = segmentation.clear_border(img)
    label = measure.label(img, connectivity=None)
    label = morphology.opening(label, selem=None, out=None)
    label = morphology.erosion(label, selem=None, out=None, shift_x=False, shift_y=False)
    label = measure.label(label, connectivity=None)
    img = morphology.remove_small_objects(label, min_size=min_size, connectivity=2)
    #img = morphology.closing(img, selem=None, out=None)
    img = morphology.closing(img, morphology.square(3))
    #img = (img < 0.5) * 1
    img = (img > 0.5) * 1
    return img


path = '../image/SK_means/liver/01.png'
#img = boundary_processing(path)

#img = morphology.opening(img, selem=None, out=None)
img = boundary_processing(path,min_size=58900)

plt.figure(figsize=(10,10))
plt.imshow(img,plt.cm.gray)
plt.show()