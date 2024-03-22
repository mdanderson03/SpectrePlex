import numpy as np
import os
from skimage import io, morphology, restoration, filters
from matplotlib import pyplot as plt
import os
from autocyplex import cycif
from pycromanager import Core
import matplotlib.pyplot as plt
import time

microscope = cycif()
core = Core()

file_directory = r'E:\19-3-24 healthy\A488\Stain\cy_3\Tiles\focused'
tiss_directory = r'E:\19-3-24 healthy\Tissue_Binary'
os.chdir(tiss_directory)
tissue = io.imread(r'x1_y_1_tissue.tif')
os.chdir(file_directory)
image = io.imread(r'x1_y_1_c_A488.tif')

csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)

def preliminaries(n, x):
  """Some math that is shared across multiple algorithms."""
  assert np.all(n >= 0)
  x = np.arange(len(n), dtype=n.dtype) if x is None else x
  assert np.all(x[1:] >= x[:-1])
  w0 = clip(csum(n))
  w1 = clip(dsum(n))
  p0 = w0 / (w0 + w1)
  p1 = w1 / (w0 + w1)
  mu0 = csum(n * x) / w0
  mu1 = dsum(n * x) / w1
  d0 = csum(n * x**2) - w0 * mu0**2
  d1 = dsum(n * x**2) - w1 * mu1**2
  return x, w0, w1, p0, p1, mu0, mu1, d0, d1

def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5):
  assert nu >= 0
  assert tau >= 0
  assert kappa >= 0
  assert omega >= 0 and omega <= 1
  x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
  v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
  v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
  f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa *      omega)  * np.log(w0)
  f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
  return argmax(x, f0 + f1), f0 + f1

image = image * tissue
indicies = np.nonzero(image)
bins = 256
hist_n, hist_edge = np.histogram(image[indicies].ravel(), bins=bins)

thresh = GHT(hist_n[0:bins], hist_edge[0:bins],nu=2E1, tau=2E5, kappa=2E2, omega=.2)[0]
print(thresh)

thresh = filters.threshold_otsu(image[indicies])
print(thresh)

x_axis = np.linspace(0,30,30)
signal_noises = np.linspace(0,10,30)
for x in range(0, 30):

    core.set_exposure(5 + x*5)

    core.snap_image()
    tagged_image = core.get_tagged_image()
    pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
    image1= np.nan_to_num(pixels, posinf=65500)
    image1 = image1.astype('int32')

    core.snap_image()
    tagged_image = core.get_tagged_image()
    pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
    image2= np.nan_to_num(pixels, posinf=65500)
    image2 = image2.astype('int32')

    hist_n, hist_edge = np.histogram(image1.ravel(), bins=bins)
    thresh = GHT(hist_n[0:bins], hist_edge[0:bins],nu=2E1, tau=2E5, kappa=2E2, omega=.2)[0]
    image1[image1 < thresh] = 0
    indicies = np.nonzero(image1)

    diff = image1 - image2
    stdev = np.std(diff[indicies])
    mean = (np.mean(image1[indicies]) + np.mean(image2[indicies]))/2

    signal_noise = mean/stdev

    signal_noises[x] = signal_noise
    x_axis[x] = 5 + x*5
#signal_noises = np.sqrt(signal_noises)
print(signal_noises)

plt.scatter(x_axis, signal_noises)
plt.show()