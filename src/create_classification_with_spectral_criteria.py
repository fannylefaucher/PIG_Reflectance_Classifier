# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:22:26 2024

@author: lefauchf
"""

#%% IMPORT PYTHON PACKAGES
import numpy as np
# import matplotlib.pyplot as plt
import os
import glob
from sklearn.decomposition import FastICA
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
import pandas as pd
from pathlib import Path

srcpath = Path(__file__).resolve().parent
# plt.close('all')
#%% LOAD DATABASE

# Load R database
df_R = pd.read_excel(srcpath.parent/'data'/'R_database.xlsx')
Wavelength = df_R['Wavelength (nm)'].to_numpy()
R = df_R.iloc[:, 1:].to_numpy()

# Load R_nowater database
df_R_nowater = pd.read_excel(srcpath.parent/'data'/'R_nowater_database.xlsx')
Wavelength_nowater = df_R_nowater['Wavelength (nm)'].to_numpy()
liste_pig_names = df_R.columns[1:].to_numpy()
R_nowater = df_R_nowater.iloc[:, 1:].to_numpy()

# R = np.loadtxt(r"data/R.txt")
# R_nowater = np.loadtxt(r"data/R_nowater.txt")
# Wavelength = np.loadtxt(r"data/Wavelength.txt")
# Wavelength_nowater = np.loadtxt(r"data/Wavelength_nowater.txt")

# f_pignames = open(r'data/liste_pigments_database.txt','r')
# liste_pig_names1 = f_pignames.readlines()
# liste_pig_names1.pop(0)
# liste_pig_names2 = []
# for name in liste_pig_names1:
#     liste_pig_names2.append(name[:-1])
# liste_pig_names = np.array(liste_pig_names2)
# f_pignames.close()

# R = np.loadtxt(r"C:\Users\lefauchf\analyse-pigments\R.txt")
# R_nowater = np.loadtxt(r"C:\Users\lefauchf\analyse-pigments\R_nowater.txt")
# Wavelength = np.loadtxt(r"C:\Users\lefauchf\analyse-pigments\Wavelength.txt")
# Wavelength_nowater = np.loadtxt(r"C:\Users\lefauchf\analyse-pigments\Wavelength_nowater.txt")

n_R = np.shape(R_nowater)[1]
n_wave = np.shape(R_nowater)[0]

# fig = plt.figure(figsize=(12, 10))
# plt.plot(Wavelength,R[:,10:20])
# plt.xlim([450,4186])
# plt.ylim([0,1])
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance factor')
# plt.title('Pigment database')
# plt.axvline(x = 1342, color = 'gray')
# plt.axvline(x = 1450, color = 'gray')
# plt.axvline(x = 1790, color = 'gray')
# plt.axvline(x = 1962, color = 'gray')
# plt.axvline(x = 2470, color = 'gray')
# plt.axvline(x = 2850, color = 'gray')
# plt.show()


#%% BAND DEPTH CRITERIA


Band_wavelength = np.array([[3978,3730,4170],[2206,2130,2230],[1910,1798,2070],[2340,2250,2366],[3226,3162,3298],[870,750,1202], [1450,1342,1550], [2322,2250,2350], [3802,3738,3834]])# pour chaque ligne contient les longueurs d'onde: de la bande, de l'epaulement gauche, de l'epaulement droit
n_crit_depth = np.shape(Band_wavelength)[0]

Band_idx =  np.zeros(np.shape(Band_wavelength)).astype(int)
for i in range(n_crit_depth): 
    for j in range (3):
        Band_idx [i,j] = int((np.abs(Wavelength_nowater - Band_wavelength[i,j])).argmin())
    
Prof_band = np.zeros((n_crit_depth,n_R))
for i in range(n_crit_depth): 
    # Prof_band[i,:] = (R_nowater[Band_idx[i,1],:]+(R_nowater[Band_idx[i,2],:]-R_nowater[Band_idx[i,1],:])/(Wavelength_nowater[Band_idx[i,2]]-Wavelength_nowater[Band_idx[i,1]])*(Wavelength_nowater[Band_idx[i,0]]-Wavelength_nowater[Band_idx[i,1]])-R_nowater[Band_idx[i,0],:])/R_nowater[Band_idx[i,0],:]
    Prof_band[i,:] = (R_nowater[Band_idx[i,1],:]+(R_nowater[Band_idx[i,2],:]-R_nowater[Band_idx[i,1],:])/(Wavelength_nowater[Band_idx[i,2]]-Wavelength_nowater[Band_idx[i,1]])*(Wavelength_nowater[Band_idx[i,0]]-Wavelength_nowater[Band_idx[i,1]])-R_nowater[Band_idx[i,0],:])/(R_nowater[Band_idx[i,1],:]+(R_nowater[Band_idx[i,2],:]-R_nowater[Band_idx[i,1],:])/(Wavelength_nowater[Band_idx[i,2]]-Wavelength_nowater[Band_idx[i,1]])*(Wavelength_nowater[Band_idx[i,0]]-Wavelength_nowater[Band_idx[i,1]]))
minmaxprofband = np.zeros((np.shape(Prof_band)[0],2))
minmaxprofband[:,0] = np.min(Prof_band,1)
minmaxprofband[:,1] = np.max(Prof_band,1)
# Prof_band[n_crit,:] = np.gradient(R_nowater, axis =0)
Prof_band_norm = ((Prof_band.T - np.min(Prof_band, axis = 1))/(np.max(Prof_band, axis = 1)- np.min(Prof_band, axis = 1))).T
#%% WAVELENGTH OF ABSORPTION BAND


sigma = 3
R_smooth = gaussian_filter(R_nowater,sigma, axes = 0)
deriv1 =  np.gradient(R_smooth, axis =0)
deriv1rough =  np.gradient(R_nowater, axis =0)
deriv2 = np.gradient(deriv1, axis =0)
deriv2rough = np.gradient(deriv1rough, axis =0)

# Crit 7
idx_lbd_min = np.zeros(n_R)
idx_lbd_min = np.argmin(np.where((deriv2 > 0) & (np.tile(Wavelength_nowater,(n_R,1)).T > 750) & (np.tile(Wavelength_nowater,(n_R,1)).T < 1000) ,np.abs(deriv1),np.inf), axis = 0)

# Crit 8
wl_kaolinite1 = 2162
kaolinite = np.where((np.abs(Wavelength_nowater[np.argmax(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-wl_kaolinite1)<15,deriv2,0), axis = 0)]-wl_kaolinite1)<5) & (np.max(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-wl_kaolinite1)<15,deriv2,0), axis = 0)>0),1,0) + np.where((np.abs(Wavelength_nowater[np.argmax(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-2206)<15,deriv2,0), axis = 0)]-2206)<5) & (np.max(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-2206)<15,deriv2,0), axis = 0)>0),1,0)#
n_crit = n_crit_depth + 2

# Crit 9
bandes_2350 = np.array(argrelextrema(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-2350)<60,deriv2,-np.inf),np.greater, order = 1, axis = 0))
bandes_2350_sort = bandes_2350[:,np.argsort(bandes_2350[1])]
bandes_2350_sort[0,:] = Wavelength_nowater[bandes_2350_sort[0,:]]
# bandes_2350_crit = np.zeros((4,n_R))
# for i in range(np.shape(bandes_2350_sort)[1]):
#     if 2310 <= bandes_2350_sort[0,i]<= 2326:
#         bandes_2350_crit[0,bandes_2350_sort[1,i]] = 1
#     if 2334 <= bandes_2350_sort[0,i]<= 2346:
#         bandes_2350_crit[1,bandes_2350_sort[1,i]] = 1
#     if 2350 <= bandes_2350_sort[0,i]<= 2354:
#         bandes_2350_crit[2,bandes_2350_sort[1,i]] = 1
#     if 2378<= bandes_2350_sort[0,i]<= 2386:
#         bandes_2350_crit[3,bandes_2350_sort[1,i]] = 1
bandes_2350_crit = np.zeros((5,n_R))
for i in range(np.shape(bandes_2350_sort)[1]):
    if 2294 <= bandes_2350_sort[0,i]<= 2306:
        bandes_2350_crit[4,bandes_2350_sort[1,i]] = 1
    if 2310 <= bandes_2350_sort[0,i]<= 2330:
        bandes_2350_crit[0,bandes_2350_sort[1,i]] = 1
    if 2334 <= bandes_2350_sort[0,i]<= 2346:
        bandes_2350_crit[1,bandes_2350_sort[1,i]] = 1
    if 2350 <= bandes_2350_sort[0,i]<= 2354:
        bandes_2350_crit[2,bandes_2350_sort[1,i]] = 1
    if 2378<= bandes_2350_sort[0,i]<= 2386:
        bandes_2350_crit[3,bandes_2350_sort[1,i]] = 1

# Crit 11
bande_2706 = np.where((deriv1rough[Wavelength_nowater==2706,:]>0) | (deriv1rough[Wavelength_nowater==2710,:]>0)| (deriv1rough[Wavelength_nowater==2714,:]>0),1,0)[0]
bande_2786 = np.where(((deriv1rough[Wavelength_nowater==2706,:]<0) & (deriv1rough[Wavelength_nowater==2710,:]<0)& (deriv1rough[Wavelength_nowater==2714,:]<0))&((deriv1[Wavelength_nowater==2778,:]>0)|(deriv1[Wavelength_nowater==2782,:]>0)|(deriv1[Wavelength_nowater==2786,:]>0)),1,0)[0]

# Crit 12
idx_lbd_1100 = np.zeros(n_R)
idx_lbd_1100 = np.argmax(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf), axis = 0)
# val_lbd_1100 = (np.max(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf), axis = 0)-np.min(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf)))/(np.max(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf))-np.min(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf)))
val_lbd_1100_valabs = np.max(np.where(np.abs(np.tile(Wavelength_nowater,(n_R,1)).T-1100)<200,deriv1,-np.inf), axis = 0)
minmax_val_lbd_1100_valabs = np.array([np.min(val_lbd_1100_valabs),np.max(val_lbd_1100_valabs)])
val_lbd_1100 = (val_lbd_1100_valabs-np.min(val_lbd_1100_valabs))/(np.max(val_lbd_1100_valabs)-np.min(val_lbd_1100_valabs))*100

# Crit 13
slope_1450_valabs = (R_nowater[np.argmin(np.abs(Wavelength_nowater-1550)),:]-R_nowater[np.argmin(np.abs(Wavelength_nowater-1342)),:])/(1550-1342)
minmax_slope_1450_valabs = np.array([np.min(slope_1450_valabs),np.max(slope_1450_valabs)])
slope_1450 = (slope_1450_valabs-np.min(slope_1450_valabs))/(np.max(slope_1450_valabs)-np.min(slope_1450_valabs))*100

#%% save criteria values for each spectrum


spectra_critvalues_table = np.hstack((liste_pig_names[:,np.newaxis],
                                      Prof_band.T,
                                      Wavelength_nowater[idx_lbd_min][:,np.newaxis],
                                      kaolinite[:,np.newaxis],
                                      bandes_2350_crit.T,
                                      bande_2706[:,np.newaxis],
                                      bande_2786[:,np.newaxis],
                                      Wavelength_nowater[idx_lbd_1100][:,np.newaxis],
                                      val_lbd_1100[:,np.newaxis],
                                      slope_1450_valabs[:,np.newaxis]
                                      ))

# np.savetxt('spectra_critvalues_table.txt', spectra_critvalues_table[:,:].astype(str), fmt='%s',delimiter=',', encoding=None)

#%% Plot stuff

# plt.imshow(bandes_2350_crit, aspect = 10)
# plt.show()

# # plt.scatter(bandes_2350_sort[1,:],bandes_2350_sort[0,:])
# # plt.show()
# # fig = plt.figure(figsize=(14, 12))
# # plt.scatter(Wavelength_nowater[bandes_2350[0,:]],bandes_2350[1,:],marker = '+')
# # plt.grid(which='major', color='gray', linestyle='-')
# # plt.grid(which='minor', color='lightgray')
# # plt.minorticks_on()
# # plt.show()



# i = 150
# # interval = [2100,2250]
# interval = [2675,2875]

# fig = plt.figure(figsize=(6, 10))
# plt.plot(Wavelength_nowater,R_nowater[:,i:i+10], label = (str(i)))#,'k'
# plt.plot(Wavelength_nowater,R_smooth[:,i:i+10])
# # plt.xlim([450,4186])
# plt.xlim(interval)
# plt.ylim([0,1])
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance factor')
# plt.title('Pigment database')
# plt.axvline(x = 1342, color = 'gray')
# plt.axvline(x = 1450, color = 'gray')
# plt.axvline(x = 1790, color = 'gray')
# plt.axvline(x = 1962, color = 'gray')
# plt.axvline(x = 2470, color = 'gray')
# plt.axvline(x = 2850, color = 'gray')
# plt.grid(which='major', color='gray', linestyle='-')
# plt.grid(which='minor', color='lightgray')
# plt.minorticks_on()
# # plt.xlim([750,1202])
# plt.show()


# fig = plt.figure(figsize=(14, 12))
# plt.axvline(x = 850, color = 'gray')
# plt.axvline(x = 750, color = 'gray')
# plt.axvline(x = 1202, color = 'gray')
# plt.grid(which='major', color='gray', linestyle='-')
# plt.grid(which='minor', color='lightgray')
# plt.minorticks_on()
# # plt.plot(Wavelength_nowater, test[:,i:i+10],'k')
# plt.plot(Wavelength_nowater,deriv1[:,i:i+10],':')
# plt.plot(Wavelength_nowater,deriv1rough[:,i:i+10],'-')
# # plt.xlim([750,1202])
# # plt.xlim([900,1342])
# plt.xlim(interval)
# # plt.xlim([2695,2710])

# k = 0.015
# plt.ylim([-k,k])
# plt.title('Deriv 1')
# plt.show()

# fig = plt.figure(figsize=(14, 12))
# plt.axvline(x = 850, color = 'gray')
# plt.axvline(x = 750, color = 'gray')
# plt.axvline(x = 1202, color = 'gray')
# plt.grid(which='major', color='gray', linestyle='-')
# plt.grid(which='minor', color='lightgray')
# plt.minorticks_on()
# # plt.plot(Wavelength_nowater, test[:,i:i+10],'k')
# plt.plot(Wavelength_nowater,deriv2[:,i:i+10],'-', label = ' ')
# plt.plot(Wavelength_nowater,deriv2rough[:,i:i+10],':')
# # plt.xlim([750,1202])
# # plt.xlim([900,1342])
# plt.xlim(interval)
# k = 0.0005
# plt.ylim([-k,k])
# plt.title('Deriv 2')
# plt.legend()
# plt.show()


# fig = plt.figure(figsize=(3, 10))
# # i = 0
# colors = plt.cm.tab10(np.linspace(0,1,10))
# style = [':','-','--']
# for k in range(i, i+10):
#     # plt.plot(Wavelength_nowater,R_smooth[:,k],c = 'k', linestyle = style[kaolinite[2]])
#     plt.plot(Wavelength_nowater,R_nowater[:,k],c = colors[k-i], linestyle = style[bande_2706[k]], label = ' ')
#     plt.axvline(x = Wavelength_nowater[idx_lbd_1100[k]], c = colors[k-i])
# # plt.plot(Wavelength_nowater,R_smooth[:,i:i+10])
# # plt.xlim([450,4186])
# plt.ylim([0,1])
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance factor')
# plt.title('Pigment database')
# plt.axvline(x = 1342, color = 'gray')
# plt.axvline(x = 1450, color = 'gray')
# plt.axvline(x = 1790, color = 'gray')
# plt.axvline(x = 1962, color = 'gray')
# plt.axvline(x = 2470, color = 'gray')
# plt.axvline(x = 2850, color = 'gray')
# plt.grid(which='major', color='gray', linestyle='-')
# plt.grid(which='minor', color='lightgray')
# plt.minorticks_on()
# plt.xlim(interval)
# plt.legend()
# plt.show()


# # fig = plt.figure(figsize=(3, 10))
# # # i = 0
# # plt.plot(Wavelength_nowater,R_nowater[:,bandes_2350_crit[3,:]==1])
# # # plt.plot(Wavelength_nowater,R_smooth[:,i:i+10])
# # # plt.xlim([450,4186])
# # plt.ylim([0,1])
# # plt.xlabel('Wavelength (nm)')
# # plt.ylabel('Reflectance factor')
# # plt.title('Pigment database')
# # plt.axvline(x = 1342, color = 'gray')
# # plt.axvline(x = 1450, color = 'gray')
# # plt.axvline(x = 1790, color = 'gray')
# # plt.axvline(x = 1962, color = 'gray')
# # plt.axvline(x = 2470, color = 'gray')
# # plt.axvline(x = 2850, color = 'gray')
# # plt.grid(which='major', color='gray', linestyle='-')
# # plt.grid(which='minor', color='lightgray')
# # plt.minorticks_on()
# # plt.xlim(interval)
# # plt.show()

# # plt.plot(idx_lbd_min,'o')
# # plt.show()


# #%% VISUALISE CRITERIA
# colors = plt.cm.tab10(np.linspace(0,1,n_crit_depth))
# fig = plt.figure(figsize=(12, 10))
# plt.plot(Wavelength,R)
# plt.xlim([450,4186])
# plt.ylim([0,1])
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance factor')
# plt.title('Pigment database')
# for i in range(n_crit_depth):
#     plt.axvline(x = Band_wavelength[i,0], color = colors[i])
#     plt.axvline(x = Band_wavelength[i,1], color = colors[i], linestyle = ':')
#     plt.axvline(x = Band_wavelength[i,2], color = colors[i], linestyle = ':')
# plt.axvline(x = 1342, color = 'gray')
# plt.axvline(x = 1450, color = 'gray')
# plt.axvline(x = 1790, color = 'gray')
# plt.axvline(x = 1962, color = 'gray')
# plt.axvline(x = 2470, color = 'gray')
# plt.axvline(x = 2850, color = 'gray')
# plt.show()

# colors = plt.cm.tab10(np.linspace(0,1,n_crit_depth))
# colors2 = plt.cm.Greys(np.linspace(0.1,0.8,n_R))
# fig = plt.figure(figsize=(8, 6))
# for i in range(n_R):
#     plt.plot(Wavelength_nowater,R_nowater[:,i],c = colors2[i])
# plt.xlim([450,4186])
# plt.ylim([0,1])
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance factor')
# plt.title('Pigment database')
# for i in range(n_crit_depth):
#     plt.axvline(x = Wavelength_nowater[Band_idx[i,0]], color = colors[i])
#     plt.axvline(x = Wavelength_nowater[Band_idx[i,1]], color = colors[i], linestyle = '--')
#     plt.axvline(x = Wavelength_nowater[Band_idx[i,2]], color = colors[i], linestyle = '--')
#     # print(Wavelength_nowater[Band_idx[i,0]],Wavelength_nowater[Band_idx[i,1]],Wavelength_nowater[Band_idx[i,2]] )
# # plt.axvline(x = 1342, color = 'gray')
# # plt.axvline(x = 1450, color = 'gray')
# # plt.axvline(x = 1790, color = 'gray')
# # plt.axvline(x = 1962, color = 'gray')
# # plt.axvline(x = 2470, color = 'gray')
# # plt.axvline(x = 2850, color = 'gray')
# plt.show()


#%% PLOT CRITERIA 2D
# for i in range(n_crit): 
#     for j in range(i+1,n_crit):
#         plt.plot(Prof_band[i,:],Prof_band[j,:],'o')
#         plt.title(' i = %s, j = %s'%(i,j))
#         plt.show()




#%% PLOT CRITERIA 3D


# ax = plt.figure().add_subplot(projection='3d')

# ax.scatter(Prof_band_norm[1,:], Prof_band_norm[2,:],Prof_band_norm[3,:], c = Prof_band_norm[4,:] )

# ax.view_init(elev=20., azim=-35, roll=0)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()


# ax = plt.figure().add_subplot(projection='3d')
# ax.scatter(Prof_band_norm[3,:], Prof_band_norm[4,:],Prof_band_norm[5,:], c = Prof_band_norm[2,:], cmap = 'inferno' )
# ax.view_init(elev=20., azim=-35, roll=0)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()


#%% ICA ON CRITERIA

# mean = np.zeros((np.shape(Prof_band_norm)[0],1))
# mean[:,0] = np.mean(Prof_band_norm, axis = 1)


# n_compo = n_crit
# transformer = FastICA(n_components=n_compo,

#         random_state=0,

#         whiten='unit-variance')

# R_transformed = transformer.fit_transform(Prof_band_norm-mean)# ICA components (spectral sources of the database)

# W = transformer.components_# weight of each ICA component in each spectrum
# #%%
# colors = plt.cm.jet(np.linspace(0,1,n_compo))
# for i in range(n_compo):
#     fig = plt.figure(figsize=(12, 10))
#     plt.subplot(2,1,1)
#     plt.scatter(Band_wavelength[:,0],np.abs(R_transformed[:,i]), color=colors[i], label = 'ICA component %s'%int(i))
#     plt.title('Independant spectral sources of the database')
#     plt.xlabel('Wavelength (nm)')
#     plt.ylabel('ICA components')
#     k=5
#     plt.ylim(0,k)
#     plt.legend()
#     plt.axhline(2, c='k')
#     plt.grid(which='major', color='gray', linestyle='-')
#     plt.grid(which='minor', color='lightgray')
#     plt.minorticks_on()

#     plt.subplot(2,1,2)
#     plt.hist(W[i,:],bins = 80, color = colors[i], label = 'ICA component %s'%int(i))
#     plt.ylabel('Number of spectra')
#     plt.xlabel('ICA value')
#     plt.ylim([0,20])
#     # plt.xlim([np.min(W),np.max(W)])
#     plt.grid(which='major', color='gray', linestyle='-')
#     plt.grid(which='minor', color='lightgray')
#     plt.minorticks_on()
#     plt.legend()
#     plt.show()
    
# Index_S_significant = np.array([0])#np.array([2,7,9,4,10,15,18,20,21,22,22])-1#np.array([14,17])-1##np.array([2,3,4,5,6,7,8,9,10,11,12,15,18,20,22,23,24,25,26,28,29])-1
# n_signif = np.size(Index_S_significant)
# S_significant = np.zeros((np.shape(R_transformed)[0],n_signif))
# W_significant = np.zeros((n_signif,np.shape(W)[1]))

# for i in range(n_signif):
#     S_significant[:,i] = R_transformed[:,Index_S_significant[i]]
#     W_significant[i,:] = W[Index_S_significant[i],:]
    
# # #%% SELECT ICA-VALUE THRESHOLD TO DEFINE ENDMEMBERS

# # # Mixing_W_signif = np.linalg.pinv(W_significant[:,:].T)

# # thresh = np.zeros(np.shape(W_significant))
# # thresh += np.array([[-1.75,2,0]]).T *10**14#np.array([[20,11.5,8.5,7,20,-12,8,20,-10,0,-26]]).T#np.array([[11,0]]).T#np.array([[12,8,20,18,12,20,10,15,30]]).T#np.array([[20,23,12,12,20,11,10,8,22,18,19,14,12,22,20,16,10,20,26,15,18]]).T#np.array([[12,17,9,12,16,10,8,7,17,14,12,10,8,10,10,8,8,18,20,13,10]]).T

# # R_reconstructed = R_nowater
# # EM_binary_0 = np.where(W_significant>thresh ,1,0)* np.where(thresh>0,1,0)+np.where(W_significant<thresh ,1,0)*np.where(thresh<0,1,0)+np.where(np.abs(W_significant-0.25*10**14)<1*10**14 ,1,0)*np.where(thresh==0,1,0)
# # EM_binary = np.zeros(np.shape(EM_binary_0))
# # EM_binary = (EM_binary_0.T/np.sum(EM_binary_0,1)).T
# # # plt.imshow(EM_binary)
# # # plt.show()
# # EM = R_reconstructed@EM_binary.T
# # EM_ICA = W_significant@EM_binary.T
# # colors = plt.cm.jet(np.linspace(0,1,n_signif))
# # k = 4
# # for i in range(n_signif):
# #     fig = plt.figure(figsize=(10, 10))
# #     plt.subplot(2,1,1)
# #     # plt.plot(Wavelength_nowater,np.abs(S_significant[:,i])/np.max(np.abs(S_significant[:,i])), color=colors[i])
# #     plt.plot(Wavelength_nowater,R_reconstructed*EM_binary_0[i,:] )
# #     plt.plot(Wavelength_nowater,EM[:,i],':',color = 'k')#color= colors[i])
# #     plt.xlabel('Wavelength (nm)')
# #     plt.ylabel('Reflectance factor')
# #     plt.title('Endmember n° %s, ICA component n° %s'%(i,Index_S_significant[i]))
# #     plt.ylim([0,1])
# #     # plt.show()
    
# #     plt.subplot(2,1,2)
# #     plt.hist(W[Index_S_significant[i],:],bins = 80, color = colors[i], label = 'ICA component %s'%int(Index_S_significant[i]))
# #     plt.ylabel('Number of spectra')
# #     plt.xlabel('ICA value')
# #     plt.ylim([0,20])
# #     plt.xlim([np.min(W),np.max(W)])
# #     plt.grid(which='major', color='gray', linestyle='-')
# #     plt.grid(which='minor', color='lightgray')
# #     plt.minorticks_on()
# #     plt.axvline(x = thresh[i][0], color = 'gray')
# #     plt.legend()
# #     plt.show()
# #%%

# n_class =  10

# W_significant_norm = (W_significant-np.min(W_significant))/(np.max(W_significant)-np.min(W_significant))

# for i in range(n_class):
#     plt.plot(Wavelength_nowater,R_nowater[:,np.abs(W_significant_norm[0,:]-(i+0.5)/n_class)<1/n_class])
#     plt.show()
    