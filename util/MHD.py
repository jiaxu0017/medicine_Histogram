import glob
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np

img_file = '/home/panbo/PycharmProjects_jiaxu/datasets/TianChi_lung/train_subset00/LKDS-00066.mhd'


itk_img = sitk.ReadImage(img_file)
img_array = sitk.GetArrayFromImage(itk_img)
#center = np.array([node_x, node_y, node_z])
