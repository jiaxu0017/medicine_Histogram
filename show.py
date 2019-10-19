import pydicom
import matplotlib.pyplot as plt
import  os
from PIL import Image
import numpy as np


path = '/home/panbo/PycharmProjects_jiaxu/datasets/Patient20190808_deeplearning/Patient20190808_deeplearning/190474/CT-Sim images'

dir = os.listdir(path)

dcm = pydicom.read_file(path + '/' + dir[2])

image = dcm.pixel_array
image =(image - image.min(axis=0))/(image.max(axis=0) - image.min(axis=0))


img = Image.fromarray(image.astype(np.int8),mode='L')

#img = np.asarray(img)

img.show()
print(dcm)
print(img.im)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()