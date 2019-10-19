from skimage import data,io,color,feature,measure
import skimage.morphology as sm
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

# # skimage.morphology.dilation
# img = data.checkerboard()
# dst1 = sm.dilation(img,sm.square(5))
# dst2 = sm.dilation(img,sm.square((15)))
# # skimage.morphology.erosion
# dst1 = sm.erosion(img,sm.square(5))
# dst2 = sm.erosion(img,sm.square(15))
# #black_tophat and white_tophat
# img = data.camera()
# img = color.rgb2grey(img)
# dst1 = sm.black_tophat(img,sm.square(20))
# dst2 = sm.white_tophat(img,sm.square(20))

# # convex_hull_image
# img = color.rgb2grey(data.horse())
# img = (img < 0.5) * 1
#
# chull = sm.convex_hull_image(img)

# img = color.rgb2grey(data.coins())
# edgs = feature.canny(img, sigma=3, low_threshold=10,high_threshold=50)
#
# chull = sm.convex_hull_object(edgs)

def microstructure(l=256):
    n = 5
    x, y = np.ogrid[0:l, 0:l]  #生成网络
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)  #随机数种子
    points = l * generator.rand(2, n**2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndi.gaussian_filter(mask, sigma=l/(4.*n)) #高斯滤波
    return mask > mask.mean()

data = microstructure(l = 128) *1

labels = measure.label(data,connectivity=2)
dst = color.label2rgb(labels)
dst = sm.remove_small_objects(labels, min_size=300,connectivity=1)
print('regions number:', labels.max()+1)

if __name__ == '__main__':
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(data,plt.cm.gray)
    plt.subplot(122)
    plt.imshow(dst,plt.cm.gray)
    # plt.subplot(133)
    # plt.imshow(dst2,plt.cm.gray)


    plt.show()