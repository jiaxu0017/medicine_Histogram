import pydicom
#import dicom
import glob
import pandas as pd
import  numpy as np
import skimage, os
from skimage import  measure, morphology
from scipy import ndimage as ndi
import matplotlib
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc

def load_scan(path):
    slices = [pydicom.read_file(path+'/'+s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] =0

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing = [1,1,1]):
    spacing = map(float, ([scan[0].SliceThickness, scan[0].PixelSpacing[0],scan[0].PixelSpacing[1]]))
    spacing =np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor,mode='nearest')

    return image, new_spacing

def plot_3d(image, threshold=-300):
    p = image.transpose(2,1,0)
    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha = 0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()

def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(50,50))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i],cmap=plt.cm.bone)

def get_segmented_lungs(im,plot=False):
    if plot == True:
        f, plots = plt.subplots(8,1,figsize=(5,40))

    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].set_title('binary image')
        plots[0].imshow(binary, cmap=plt.cm.bone)

    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].set_title('after clear border')
        plots[1].imshow(cleared,cmap=plt.cm.bone)

    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].set_title('found all connective graph')
        plots[2].imshow(label_image,cmap=plt.cm.bone)

    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].set_title(' Keep the labels with 2 largest areas')
        plots[3].imshow(binary, cmap=plt.cm.bone)

    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].set_title('seqerate the lung nodules attacthed to the blood vessels')
        plots[4].imshow(binary,cmap=plt.cm.bone)

    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].set_title('keep nodules attached to the lung wall')
        plots[5].imshow(binary,cmap=plt.cm.bone)

    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].set_title('Fill in the small holes inside the binary mask of lungs')
        plots[6].imshow(binary, cmap=plt.cm.bone)

    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].set_title('Superimpose the binary mask on the input image')
        plots[7].imshow(binary, cmap=plt.cm.bone)

    return im

def largest_label_volume(im,bg=-1):
    vals, counts = np.unique(im,return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures = True):
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    backgroung_label = labels[0,0,0]

    binary_image[backgroung_label == labels] = 2

    if fill_lung_structures:
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice -1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg =0)

            if l_max is not None:
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1
    binary_image = 1 - binary_image

    labels =measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:
        binary_image[labels != l_max] = 0

    return binary_image

MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1
    image[image < 0] = 0
    return image
PIXEL_MEAN = 0.25
def zerp_center(image):
    image = image - PIXEL_MEAN
    return image

def testprint(x):
    print("*********************")
    print('test:',x)
    print('*********************')


img_path = '/home/panbo/PycharmProjects_jiaxu/datasets/DSB3/stage1/stage1/ffe02fe7d2223743f7fb455dfaff3842'
frist_patient = load_scan(img_path)
frist_patient_pixels = get_pixels_hu(frist_patient)
pix_resampled, spacing = resample(frist_patient_pixels,frist_patient,[1,1,1])
print(spacing)
print("**************")
print(pix_resampled)
plot_3d(pix_resampled,400)
#plot_ct_scan(pix_resampled)
#get_segmented_lungs(pix_resampled[10],True)
#