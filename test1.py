import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
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

print(numpyImage.shape)

plt.figure(figsize=(10,10))
plt.imshow(numpyImage[19])
plt.show()

print(numpyOrgin)
print(numpySpacing)

# slice = 120
# image = np.squeeze(numpyImage[slice])
# plt.imshow(image,cmap='gray')
# plt.show()

def readCSV(filename):
    lines = []
    with open(filename,'r') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

#anno_path = '~/home/panbo/PycharmProjects_jiaxu/datasets/TianChi_lung/csv/train/annotations.csv'
annos = readCSV(anno_path)
print(len(annos))
print(annos[0:3])

cand = annos[24]
print(cand)

worldCoord = np.asarray([float(cand[1]),float(cand[2]),float(cand[3])])
voxelCoord = worldToVoxelCoord(worldCoord, numpyOrgin, numpySpacing)
print(voxelCoord)