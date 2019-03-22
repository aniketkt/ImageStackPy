#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:18:07 2019

@author: atekawade

This module provides a set of functions for a segmentation algorithm using a single-feature gaussian mixture model.
The feature is voxel intensity across the volume. Hence, spatial / contextual information is not used in the segmentation process.
However, it works great in many situations. This algorithm is inspired by Jake VanderPlas' example of 1D GMM found here:
http://www.astroml.org/book_figures/chapter4/fig_GMM_1D.html


The general procedure is as follows
from ImageStackPy import ImageProcessing as IP
from ImageStackPy import GMM_Segmenter3D as SG
from ImageStackPy import Img_Viewer as VIEW

#Load the volume "R" as a 3D numpy array. Then slice the volume to downsize it as:
r = R[::step_size,::step_size,::step_size]


#Explore the voxel intensity histogram by orienting the volume in different ways such as:
VIEW.plot_histogram(np.swapaxes(r,0,1))
VIEW.plot_histogram(r)

#Let's say you decide visually that you see 7 components in the gaussian mixture, fit the GMM for at least 5 components:
#Notice that we have removed zeroes from the fit. This is not required.
#But in some cases, there are zero value pixels on the boundaries of a volume because you may have cropped or rotated.

N_list, BIC, models = SG.GMM_test(r[r!=0].reshape(-1,1), n_models = 5)

#View how the models fit for various N's (where N is the number of components assumed in the GMM). [4,6] is the range you want to see.
SG.show_modelfit(r[r!=0].reshape(-1,1), models, N_plots = [2,4])

#View the BIC plot
SG.show_BICplot(N_list, BIC)

# When you are ready, run the segmenter. It will automatically save tiff stacks for the segmented data if you pass argument "SaveDir"
# Note that if SaveDir argument is passed, nothing is return to conserve RAM.
C = run_segmenter(R, N = 8, n_high = 3, thresh_proba = 0.95, n_chunks = 4)


# Eliminate speckles, if needed
C = IP.XYZ_medianBlur(C, X_kern_size = 5, Y_kern_size = 5, Z_kern_size = 5)
    

"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
from ImageStackPy import ImageProcessing as IP
from ImageStackPy import Img_Viewer as VIEW
import pandas as pd
from matplotlib import patches
from scipy.ndimage.filters import gaussian_filter as gaussianBlur
from sklearn.mixture import GaussianMixture as GMM
import time
from tifffile import TiffFile



#%%
def GMM_test(data, n_models = None):
    
    N = np.arange(1,n_models+1)
    models = [None for i in range(len(N))]
    
    models = IP.Parallelize(N,get_GMM, data = data, procs = 48)#len(N))
    #for i in range(len(N)):
    #    models[i] = GMM(N[i]).fit(X)
    
    BIC = IP.Parallelize(models,get_BIC, data = data, procs = 48)#len(models))#[m.bic(data) for m in models]
    return N, BIC, models

def get_GMM(N, data = None):
    return GMM(N).fit(data.reshape(-1,1))

def get_BIC(model, data = None):
    return model.bic(data)


def show_BICplot(N_list, BIC):
    plt.figure(figsize = (4,4))
    plt.subplots_adjust(bottom = 0.15, left = 0.2)
    dBIC = BIC - np.min(BIC)
    plt.bar(N_list,dBIC/1e5)
    plt.xlabel("'N' Mixtures in Model", fontsize = 12)
    plt.ylabel("($\Delta BIC$) (x 1e5)", fontsize = 12)
    plt.ylim([0,3*dBIC[1]/1e5])
    plt.title("Bayesian Information Criterion", fontsize = 12)
    #plt.savefig(os.path.join(Plots_Dir, "BIC_plot.png"))

def show_modelfit(data, models, N_plots = None):
    
    if N_plots == None:
        N_min, N_max = 1, len(models)
    else:
        N_min, N_max = N_plots[0], N_plots[1]
    fig, ax = plt.subplots(N_max- N_min + 1,1,figsize = (6,6))
    fig.text(0.06, 0.5, "Voxel Count", va='center', rotation='vertical', fontsize = 12)
    fig.text(0.5,  0.9, "'N' Gaussian Mixture Model", fontsize = 14, ha = 'center')
    for ii in range(len(ax)):
        
        N = ii + N_min # skip N = 1
        
        h = ax[ii].hist(data, bins = 500, normed = True, alpha = 0.4)
        x= h[1]
        if ii != len(ax)-1:
            ax[ii].set_xticks([])
        else:
            ax[ii].set_xlabel("Voxel Intensity", fontsize = 12)
        ax[ii].set_yticks([])

        ax[ii].text(0.5,0.5*max(h[0]),"N = %i"%N, fontsize = 12)
        
        responsibilities = models[N-1].predict_proba(x.reshape(-1,1))
        logprob = models[N-1].score_samples(x.reshape(-1,1))
        
        #logprob, responsibilities = models[N-1].eval(x)
        pdf = np.exp(logprob)
        pdf_individual = responsibilities*pdf[:, np.newaxis]
        ax[ii].plot(x, pdf, '--k')
        ax[ii].plot(x, pdf_individual, '--')
        
def segment_(r, model = None, n_highest=1, thresh_proba = 0.9):
    
    p = get_proba(r, model, n_highest = n_highest)
    c = np.zeros_like(p)
    c[p > 0.9] = 1.0
    return c

def get_proba(r, model, n_highest=1):
    
    idx = np.argsort(model.means_[:,0])
    idx = idx[-n_highest:]
    p = model.predict_proba(r.reshape(-1,1))[:,idx]
    p = np.sum(p, axis = 1)
    p = p.reshape(r.shape)
    
    return p
    

    
        
def visualize_slice(r, s):
    
    fig, ax = plt.subplots(1, 2, figsize = (6,3))
    ax[0].axis('off')
    ax[0].imshow(r)
    ax[1].axis('off')
    ax[1].imshow(s)
       
    
    
    
def process_(r, N_list = None):
    
    models = [get_GMM(N, data = r.reshape(-1,1)) for N in N_list]
    
    return segment_(r, model = models[0], n_highest = 1) * segment_(r, models[1], n_highest = 2)


def make_chunks(R, n_chunks):
    
    chunk_size = int(np.ceil(R.shape[0]/n_chunks))
    n_last = R.shape[0] % chunk_size
    
    if n_last > 0:
        C = np.split(R[:-n_last], n_chunks - 1, axis = 0)
        C.append(R[-n_last:])
    else:
        C = np.split(R, n_chunks, axis = 0)
    
    return C


def run_segmenter(R, models, N = 2, n_high = 1, thresh_proba = 0.9, n_chunks = 1, SaveDir = None):
    
    t0 = time.time()
    
    def wrapper(R_in):
        return IP.Parallelize(IP.toStack(R_in),
                               segment_,
                               model = models[N-1],
                               n_highest = n_high,
                               thresh_proba = thresh_proba)

        
    R_chunk = make_chunks(R, n_chunks)
    
    print("Working on %i chunks of data shape " + str(R_chunk[0].shape))
    for ii in range(n_chunks):
        print("Chunk # %i"%(ii+1))
        C_chunk = wrapper(R_chunk[ii])
        
        if ii == 0:
            C = np.copy(C_chunk)
            increment_flag = False
        else:
            C = np.concatenate((C,C_chunk), axis = 0)
            increment_flag = True
    
        if SaveDir:
            IP.save_stack(C_chunk,
                          SaveDir = SaveDir,
                          increment_flag = increment_flag,
                          suffix_len = len(str(R.shape[0])))
            C = np.copy(C_chunk)
        
        del C_chunk
   
    print("Took %.3f minutes"%((time.time()-t0)/60.0))
    if not SaveDir:
        C = IP.to8bit(C, method = 'clamp')    
        return IP.toArray(C)
    else:
        return
    
    
    
    
    


    






#%%

if __name__ == "__main__":

    
    print("Segmenter based of voxel intensity fitted to a 1D Gaussian Mixture Model")
    
