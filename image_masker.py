from photutils.segmentation import detect_threshold
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.io import ascii
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.segmentation import detect_sources
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import make_source_mask

folder_path='C:/Users/Will Hamey/OneDrive/Uni/4th Year/Project/sdss_images'
write_path='C:/Users/Will Hamey/OneDrive/Uni/4th Year/Project/masked_sdss_images'
name_array = os.listdir(folder_path)

centre_mask=np.ones((60,60))
centre_mask=1-np.pad(centre_mask,70)

for i in range(len(name_array)):
  file_path=folder_path+'/'+name_array[i]
  hdu = fits.open(file_path,memmap=False)
  data = hdu[0].data
  
  mask = make_source_mask(data, mask=centre_mask, nsigma=5, npixels=10, dilate_size=11)
  mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
  mask=mask*centre_mask
  masked_data=data-data*mask+mean*mask

  name=name_array[i]
  name=name[:-6]+'G'+name[-5:]
  write_filename = write_path+"/"+name
  hdu.writeto(write_filename, overwrite=True)
