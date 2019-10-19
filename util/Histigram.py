import  SimpleITK as sitk
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

img_path = '/home/panbo/PycharmProjects_jiaxu/datasets/TianChi_lung/train_subset00/LKDS-00001.mhd'
anno_path = '/home/panbo/PycharmProjects_jiaxu/datasets/TianChi_lung/csv/train/annotations.csv'


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)  # CT numpy image
    numpyOrigin = np.array(list(itkimage.GetOrigin()))  # CT origin coordinates
    numpySpacing = np.array(list(itkimage.GetSpacing())) # CT pixel interval
    return numpyImage,numpyOrigin, numpySpacing

numpyImage, numpyOrgin, numpySpacing = load_itk_image(img_path)

def show_pixels_Histogran(numpyImage):
    print(numpyImage.flatten())
    print(numpyImage.flatten().shape)
    plt.figure(figsize=(10,10))
    plt.hist(numpyImage.flatten(),bins=80, color='c')
    plt.show()

show_pixels_Histogran(numpyImage)