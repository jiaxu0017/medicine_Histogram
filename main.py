import pydicom
import numpy as np
import pandas as pd
#import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


file = '/home/panbo/PycharmProjects_jiaxu/datasets/Patient20190808_deeplearning/Patient20190808_deeplearning/190474/'
#file = '/home/panbo/PycharmProjects_jiaxu/datasets/Patient20190808_deeplearning/CBCT_CTSIM/CT_SIM'




def patitents(file):
    patitent = os.listdir(file)
    #print(patitent.sort())
    return patitent


def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # 转换为int16，int16是ok的，因为所有的数值都应该 <32k
    image = image.astype(np.int16)

    # 设置边界外的元素为0
    image[image == -2000] = 0

    # 转换为HU单位
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)



#CBCT = pydicom.dcmread(file_CBCT)
#CTSIM = pydicom.dcmread(file_CTSIM)


IMAGE = patitents(file)
CTSIM = IMAGE[0]
CBCT = IMAGE[1]
print(CBCT)
first_patient = load_scan(file + CBCT)
print(first_patient)
first_patient_pixels = get_pixels_hu(first_patient)
plt.figure(figsize=(10,10))
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()


first_patient = load_scan(file + CTSIM)
first_patient_pixels = get_pixels_hu(first_patient)
plt.figure(figsize=(10,10))
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)  ****")
plt.ylabel("Frequency")
plt.show()
#plt.imshow(first_patient_pixels[10], cmap=plt.cm.gray)
plt.show()

# plt.figure(figsize=(10,10))
# plt.imshow(CBCT.pixel_array,cmap=plt.cm.bone)
# plt.show()