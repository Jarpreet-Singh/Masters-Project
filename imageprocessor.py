# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:39:29 2021

@author: Will Hamey
"""
import os
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import numpy as np 
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve_fft
#Change these to modify process
folder_path =    "C:/Users/Will Hamey/OneDrive/Uni/4th Year/Project/fits"
write_path =    "C:/Users/Will Hamey/OneDrive/Uni/4th Year/Project/simimg"
cropsize=200 #image dimentions
telescope_resolution = 1.4 #seeing in g band
mean= 0.0015 #Mean backgrouund level for g band
stand_dev=0.0286#standard deviation for g band
flip_fraction=3 #flip 1/3 of images

#Get file list
file_name_array = os.listdir(folder_path)
length=len(file_name_array)
for i in range(length):
    #Open fits file
    print(file_name_array[i])
    file_path=folder_path +"/"+ file_name_array[i]
    hdu = fits.open(file_path)[0]
    wcs = WCS(hdu.header)
    #reszie image 
    position = (round(hdu.data.shape[1]/2), round(hdu.data.shape[1]/2))
    size = (cropsize, cropsize)     # pixels
    if hdu.data[0].shape[1]>cropsize:
        cutout = Cutout2D(hdu.data[0], position, size)
        hdu.data = cutout.data
    else:
        pad=np.pad(hdu.data[0],cropsize,  mode='constant')
        position = (round(pad.shape[1]/2), round(pad.shape[1]/2))
        cutout = Cutout2D(pad, position, size)
        hdu.data = cutout.data
    #convulve with gaussian
    scale=hdu.header["PIXSCALE"] # Pixel size in arcsec
    sigma =telescope_resolution/scale
    psf = Gaussian2DKernel(sigma)
    convolved_image = convolve_fft(hdu.data, psf, boundary='wrap')
    hdu.data=convolved_image
    #offset the images and add sky noise
    newdata=np.empty(size)
    dx=int(np.random.randint(-5,5)*cropsize/100)
    dy=int(np.random.randint(-5,5)*cropsize/100)
    flip=np.random.randint(1,flip_fraction)
    for j in range(cropsize):
        for  k in range(cropsize):
            if 0<=j+dx<200 and 0<=k+dy<200:
                newdata[j][k]=hdu.data[j+dx][k+dy] + np.random.normal(mean,stand_dev)
            elif 0<=j+dx<200:
                newdata[j][k]=hdu.data[j+dx][k] + np.random.normal(mean,stand_dev)
            elif 0<=k+dy<200:
               newdata[j][k]=hdu.data[j][k+dy] + np.random.normal(mean,stand_dev)
            else:
                 newdata[j][k]= np.random.normal(mean,stand_dev)
    hdu.data=newdata
    #flip a fraction of images
    if flip==1:
        newdata=np.empty(size)
        for j in range(cropsize):
            for k in range(cropsize):
                newdata[k][j]=hdu.data[j][k]
        hdu.data=newdata
    #save modified image
    write_filename = write_path+"/"+file_name_array[i][10:]
    hdu.writeto(write_filename, overwrite=True)
        