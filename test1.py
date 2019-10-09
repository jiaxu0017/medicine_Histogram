import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt


img_path = '/home/panbo/PycharmProjects_jiaxu/datasets/TianChi_lung/train_subset00/LKDS-00001.mhd'
annotations = '/home/panbo/PycharmProjects_jiaxu/datasets/TianChi_lung/csv/train/annotations.csv'


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(itkimage.GetOrigin()))  # CT origin coordinates
    numpySpacing = np.array(list(itkimage.GetSpacing())) # CT pixel interval
    return numpyImage,numpyOrigin, numpySpacing

numpyImage, numpyOrgin, numpySpacing = load_itk_image(img_path)

print(numpyImage.shape)
print(numpyOrgin)
print(numpySpacing)
