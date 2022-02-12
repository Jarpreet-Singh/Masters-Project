# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 09:49:03 2021

@author: Will Hamey
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from astropy.io import fits
from astropy.io import ascii
from numpy import savetxt

#Important paths
table_path = 'SDSS_image_table.fits'
images_path = '/c4/wfh842/sdss_images'
model_path =  'CNNmodelV1.10' 
write_path = '/home/wfh842'  

#Load table and model
hdul = fits.open(table_path)  # open a FITS file
table = hdul[1].data  # assume the first extension is a table
model = tf.keras.models.load_model(model_path)
model.summary()


#Create dataset
file_name_array = os.listdir(images_path)
length=len(file_name_array)
test_data=[]
test_labels=[]
test_ID=[]
test_M=[]
test_sSFR=[]
for i in range(length):
  file_path=images_path +"/"+ file_name_array[i]
  hdu = fits.open(file_path,memmap=False)
  data = hdu[0].data
  ID=int(file_name_array[i][:-19])
  print(str(i+1)+'/'+str(length))
  for j in range(len(table)):
    if round(table[j][1])==ID:
      SFR=10**table[j][15]
      M=10**table[j][14]              
      sSFR=SFR/M
      label=0
      if sSFR>=10**-11:
        label=1
      test_ID.append(ID)
      test_M.append(M)
      test_sSFR.append(sSFR)
      test_data.append(data)
      test_labels.append(label)
      break
  hdu.close()
  del data
test_ds = tf.data.Dataset.from_tensor_slices((np.array(test_data), np.array(test_labels)))
class_names = ('Below','Above')
print('Images loaded')

#Run test data
correct=0
redx=[]
redy=[]
greenx=[]
greeny=[]
for i in range(len(test_data)):
  img_array = np.array(test_data[i])
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  ID=test_ID[i]
  M=test_M[i]
  sSFR=test_sSFR[i]
  if test_labels[i]==1:
    imgclass='Above'
  else:
    imgclass='Below'
  print(
      "Prediction: {} with a {:.2f} percent confidence, Actual= {}"
      .format(class_names[np.argmax(score)], 100 * np.max(score),imgclass)
  )
  if class_names[np.argmax(score)]==imgclass:
    correct=correct+1
    greenx.append(M)
    greeny.append(sSFR)
  else:
    redx.append(M)
    redy.append(sSFR)
print("Score="+str(correct)+'/'+str(len(test_ID)))

#Plot test results
fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(greenx,greeny,'g.',label='Correct')
ax.plot(redx,redy,'r+',label='Incorrect')
ax.axhline(y=10**-11, color='k', linestyle='-')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Stellar Mass M_0')
ax.set_ylabel('SSFR yr^-1')
ax.legend()
fig.savefig(write_path+'/TestResults.png')

f = open(write_path+"/score.txt", "w")
f.write("Score="+str(correct)+'/'+str(len(test_ID)))
f.close()



