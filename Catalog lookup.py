# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:06:49 2021

@author: Will Hamey
"""
import requests
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from astropy.io import ascii

def get(path, params=None):
     # make HTTP GET request to path
     headers = {"api-key":"d49e6e7be5bd180d4d0607bef8d6c33f"}
     r = requests.get(path, params=params, headers=headers)
     # raise exception if response code is not HTTP SUCCESS (200)
     r.raise_for_status()
     if r.headers['content-type'] == 'application/json':
         return r.json() # parse json responses automatically
     if 'content-disposition' in r.headers:
         filename = r.headers['content-disposition'].split("filename=")[1]
         with open(filename, 'wb') as f:
             f.write(r.content)
         return filename # return the filename string

     return r
folder_path =    "C:/Users/Will Hamey/OneDrive/Uni/4th Year/Project/simimg"
file_name_array = os.listdir(folder_path)
length=len(file_name_array)
catalog= np.zeros((length,8))
for i in range(length):
    ID = file_name_array[i][:-5]
    print(ID)
    url = "http://www.tng-project.org/api/TNG100-1/snapshots/95/subhalos/"+str(ID)+"/"
    r = get(url)
    label=0
    if (r['sfr']/(r['mass_stars']*10**10))>=10**-11:
        label=1
    catalog[i]=[ID,r['mass_stars']*10**10,r['sfr'],r['sfr']/(r['mass_stars']*10**10),r['mass']*10**10,r['halfmassrad'],r['halfmassrad_stars'],label]
t = QTable(catalog, names=('ID', 'M_s', 'SFR', 'SSFR','M','halfmassrad','halfmassrad_s', 'Label'), meta={'name': 'Test Table'})
ascii.write(t, 'testcatalog.csv', overwrite=True)
#to read t=ascii.read('testcatalog.csv')
'''
Plotting:
t=ascii.read('testcatalog.csv')
fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(t['Mass'],t['SFR'],'r.')
ax.set_xscale('log')
ax.set_yscale('log')
'''
