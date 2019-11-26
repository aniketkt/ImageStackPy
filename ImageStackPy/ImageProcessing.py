#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 16:33:32 2018

@author: atekawade

This is a collection of functions that perform image processing operations on stacks of 16 bit images.
Function in this file include filters, image arithmetic calculations, normalization, etc.
ImageJ is great for image processing. The motivation behind writing these functions is to be able to automate image processing procedures on large data sets using python.
The filters available in this library are built from scipy, scikit-image or opencv and are parallelized using multiprocessing library.
Credits: Dr. Andy Swantek and Dr. Brandon Sforzo contributed many essential ideas that went into the code.

Edits:
3/5/2019: Changed the backend for get_stack() to skimage instead of matplotlib to allow for reading 32 bit as well as 16 bit images.

"""



# Save / Read images stuff
from tifffile import imsave
import glob

# Math stuff
import cv2
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.ndimage.filters as scipyfilters
from skimage.feature import match_template
from skimage.feature import canny
import skimage.filters as skimagefilters
import matplotlib.pyplot as plt
import numpy.random as rd
from skimage import io as skimage_io


# For parallel processing
import os
import multiprocessing
import time
from multiprocessing import Pool
import functools
import shutil
#from joblib import Parallel, delayed



#GUI stuff (standard library)
import tkinter as TK
from tkinter import filedialog as tkFileDialog


# Some variables we will use
FLOAT32 = 'float 32 data type'
INT16 = '16 bit integer data type'
INT8 = '8 bit integer data type'

BORDER = '\n' + '#'*70
DEBUG_MODE = False # Suppress print input / output



def message(message_str):
    
    if DEBUG_MODE == True:
        print(message_str) 
    return
        
def error_message(message_str):
    
    print(message_str) 
    return

        
def get_filepath():
    root = TK.Tk()
    root.withdraw()
    ImDir = tkFileDialog.askdirectory(title = 'Select the parent analysis directory.')
    return ImDir

def get_image(userfilepath = None):
    
    if userfilepath:
        return skimage_io.imread(userfilepath)
    else:
        return

def get_stack(userfilepath = '', procs = None, nImages = None, fromto = None):

    message(BORDER)
    message("\nReading %s images from disk..."%(nImages if nImages != -1 else 'all'))
        
    t0 = time.time()
    if not userfilepath:
        error_message("ERROR: File path is required.")
        return []

    if procs == None:
        procs = multiprocessing.cpu_count()

        
    ImgFileList = sorted(glob.glob(userfilepath+'/*.tif'))
    
    if not ImgFileList: ImgFileList = sorted(glob.glob(userfilepath+'/*.tiff'))
    
    if fromto != None:
        ImgFileList = ImgFileList[fromto[0]:fromto[1]+1]
    elif nImages != None:
        ImgFileList = ImgFileList[:nImages]
    
    Im_Stack = Parallelize(ImgFileList, skimage_io.imread, procs = procs)

    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack

def get_stack_serial(userfilepath, nImages = -1):
    
    message("\nReading %s images from disk..."%(nImages if nImages != -1 else 'all'))
        
    t0 = time.time()
    if not userfilepath:
        error_message("ERROR: File path is required.")
        return []
        
    ImgFileList = sorted(glob.glob(userfilepath+'/*.tif'))
    
    nImages = len(ImgFileList) if nImages == -1 else nImages
    
    Im_Stack = [plt.imread(ImgFileList[ii]) for ii in range(nImages)]

    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack

def get_random_stack(X_size = 100, Y_size = 100, Z_size = 250):
    # Because sometimes, you just need a random stack to work things out.
    Im_Stack = rd.randn(Z_size,Y_size,X_size)
    Im_Stack = [Im_Stack[iS] for iS in range(Im_Stack.shape[0])]
    return Im_Stack



def save_stack(Im_Stack, SaveDir = '', increment_flag = False, suffix_len = None, dtype = INT16):
    
    
    message(BORDER)
    message("\nSaving %i images to directory..."%(len(Im_Stack)))
    t0 = time.time()
    if not SaveDir:
        error_message("ERROR: Save directory argument not passed. Nothing was saved.")
        return
    
    if dtype == INT8:
        Im_Stack = to8bit(Im_Stack)
    elif dtype == INT16:
        Im_Stack = to16bit(Im_Stack)
    elif dtype == FLOAT32:
        Im_Stack = tofloat32(Im_Stack)
    else:
        raise ValueError("Data type requested not recognized.")

    
    
    if not suffix_len:
        if increment_flag:
            message("ERROR: suffix_len argument required if increment_flag is True.")
            return
        else:
            suffix_len = len(str(len(Im_Stack)))
    
    last_num = 0
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)
    else:
        if not increment_flag:
            shutil.rmtree(SaveDir)
            os.makedirs(SaveDir)
        else:
            ImgFileList = sorted(glob.glob(SaveDir+'/*.tif'))
            if not ImgFileList: ImgFileList = sorted(glob.glob(SaveDir+'/*.tiff'))
            if not ImgFileList:
                last_num = 0
            else:
                last_num = int(ImgFileList[-1].split('.')[0][-suffix_len:])
            
    BaseDirName = os.path.basename(os.path.normpath(SaveDir))
    
    for iS, Img in enumerate(Im_Stack):
        img_num = str(iS+1+last_num).zfill(suffix_len)
        imsave(SaveDir + '/' + BaseDirName + img_num + '.tif', Img)

    t1 = time.time()
    
    message("\tDone in %f seconds."%(t1-t0))

    return






"""
def save_stack(Im_Stack, SaveDir = ''): #Pass LogFileName as savedir for PyNM_FULLAUTO
    
    message(BORDER)
    message("\nSaving %i images to directory..."%(len(Im_Stack)))
    t0 = time.time()
    if not SaveDir:
        error_message("ERROR: Save directory argument not passed. Nothing was saved.")
        return
    
    Im_Stack = to16bit(Im_Stack)
    
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)
    
    BaseDirName = os.path.basename(os.path.normpath(SaveDir))
    
    suffix_len = len(str(len(Im_Stack)))
    for iS, Img in enumerate(Im_Stack):
        img_num = str(iS+1).zfill(suffix_len)
        imsave(SaveDir + '/' + BaseDirName + img_num + '.tif', Img)

    t1 = time.time()
    
    message("\tDone in %f seconds."%(t1-t0))

    return
"""


def save_image(Img, SaveFileName = '', dtype = None):

    message(BORDER)
    message("\nSaving image...")
    t0 = time.time()
    if not SaveFileName:
        error_message("ERROR: Save directory argument not passed. Nothing was saved.")
        return
    
    Img = toStack(Img)
    if dtype == INT16:
        Img = to16bit(Img)
    elif dtype == INT8:
        Img = to8bit(Img)
    elif dtype = FLOAT32:
        Img = tofloat32(Img)
        
    Img = Img[0]
    imsave(SaveFileName + '.tif', Img)

    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))

    return




#####################################################################

# FUNCTIONS FOR PROBING IMAGES / DIAGNOSTICS - FOR 16 BIT IMAGES ONLY

#####################################################################

def get_ImgHist(Im_Stack, bins, plot_flag = True):
    
    Im_Stack = np.copy(toStack(Im_Stack))
    
    hist, bins = np.histogram(np.asarray(Im_Stack).ravel(), bins = bins)
    
    if plot_flag == True:    
        plt.plot(hist)
        plt.xlabel("Pixel value / Intensity (i)")
        plt.ylabel("Number of pixels")
        plt.title("Histogram")
    return hist

def get_cumImgHist(hist, plot_flag = True):
    
    cum_hist = np.cumsum(hist)
    
    if plot_flag == True:
        
        plt.plot(cum_hist)
        plt.xlabel("Intensity")
        plt.ylabel("Pixel value / Intensity (i)")
        plt.title("Cumulative Histogram")

    return cum_hist
    
def calc_STDEV_Image(Im_Stack, ret_type = FLOAT32):
    
    Img = toStack(np.std(np.float32(np.asarray(Im_Stack)), axis = 0))
    
    if ret_type == INT16:
        return to16bit(Img)
    elif ret_type == FLOAT32:
        return Img


def calc_MEAN_Image(Im_Stack, ret_type = FLOAT32):
    
    Img = toStack(np.mean(np.float32(np.asarray(Im_Stack)), axis = 0))
    
    if ret_type == INT16:
        return to16bit(Img)
    elif ret_type == FLOAT32:
        return Img
        

def calc_MEDIAN_Image(Im_Stack, ret_type = INT16):
    
    Img = toStack(np.median(np.float32(np.asarray(Im_Stack)), axis = 0))
    
    if ret_type == INT16:
        return to16bit(Img)
    elif ret_type == FLOAT32:
        return Img
    
def get_MotionTrace(Im_Stack, ROIY = np.array([0,0]), ROIX = np.array([0,0]), LogFileName = ''):
    
    t_start = time.time()
    Im_Stack = np.asarray(Im_Stack)
    if np.any(ROIY) and np.any(ROIX):
        Im_Stack = Im_Stack[:,ROIY[0]:ROIY[1],ROIX[0]:ROIX[1]]
    
    Trace = Parallelize(Im_Stack, calc_templatematch, template = Im_Stack[0])
    Trace = - np.asarray(Trace)
    Trace = np.nan_to_num((Trace - Trace.min())/(Trace.max() - Trace.min()))
    
    if LogFileName != '':
        plt.plot(Trace)
        plt.xlabel('Frame #')
        plt.ylabel('Motion Trace w.r.t Frame #1')
        plt.savefig(LogFileName + '_MotionTrace' + '.png')
        plt.close()
        message("\nDEBUG NOTE: Motion Trace file saved to: " + LogFileName + '_MotionTrace' + '.png')
        
    return Trace

def calc_templatematch(Img, template = []):
    
    return match_template(Img, template)[0,0] if np.any(template) else 0
    
def plot_profile(Im_Stack, axis = 0, coordinate = None, LogFileName = ''):
    
    Im_Stack = toStack(Im_Stack)
    
    if axis == 0:
        I = [Im_Stack[iS][:,coordinate] for iS in range(len(Im_Stack))]
    elif axis == 1:
        I = [Im_Stack[iS][coordinate, :] for iS in range(len(Im_Stack))]

    if LogFileName != '':
        plt.plot(I)
        plt.xlabel('Plotting axis')
        plt.ylabel('Intensity')
        plt.savefig(LogFileName + '_PlotProfile' + '.png')
        plt.close()
        message("\nDEBUG NOTE: Motion Trace file saved to: " + LogFileName + '_PlotProfile' + '.png')
        
    return I
      
def get_profile(Im_Stack, axis = None, X = None, Y = None, Z = None, LogFileName = ''):
    
    Im_Stack = np.asarray(toStack(Im_Stack))
    
    X = X if (type(X) == int and X <= Im_Stack.shape[2]) or X == None else min(Im_Stack.shape[2], int(X*(Im_Stack.shape[2] - 1)))
    Y = Y if (type(Y) == int and Y <= Im_Stack.shape[1]) or Y == None else min(Im_Stack.shape[1], int(Y*(Im_Stack.shape[1] - 1)))
    Z = Z if (type(Z) == int and Z <= Im_Stack.shape[0]) or Z == None else min(Im_Stack.shape[0], int(Z*(Im_Stack.shape[0] - 1)))
    
    if axis == 'Z' or axis == 0:
        I = Im_Stack[:,Y,X]
    elif axis == 'Y' or axis == 1:
        I = Im_Stack[Z,:,X]
    elif axis == 'X' or axis == 2:
        I = Im_Stack[Z,Y,:]
    else:
        error_message("ERROR: axis argument not recognized")
        I = []
    return I
        
    
    
    




###################################################################
    
# FUNCTIONS FOR STACK MANIPULATION
    
###################################################################

def get_EnsembleHyperStack(Im_Stack, FPE, ret_type = INT16):
    
    
    Im_Stack = toStack(Im_Stack)
    
    numFullEvents = int(len(Im_Stack)//FPE)
    message(BORDER)
    message("\nCalculating mean stack from %i events with %i frames per event"%(numFullEvents, FPE))

    t0 = time.time()

    # Numpy FTW! First slice the stack to eliminate incomplete events
    Im_Stack = np.asarray(Im_Stack[:numFullEvents*FPE])
    # Then slice the stack using start:stop:step as ii::FPE. Start at 'ii'th frame, then step FPE frames to next identical frame
    H = np.asarray([Im_Stack[ii::FPE,:,:] for ii in range(FPE)])
    # Now you get a 4D array with dim0 as time (length = FPE), dim1 as event (length = numFullEvents), dim2, dim3 as Y, X
    H = np.swapaxes(H,0,1)
    
    # Done. This is done 10 times faster than the best for loop I could come up with.
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))

    return H





def toStack(Im_Stack):

    # Makes Im_Stack a "stack" or python list of images, each image being a 2D np array
    # Im_Stack has the shape Z,Y,X
    # This function checks if Im_Stack is passed alternately passed as (1) single image or (2) 3D np array with dims as Z,Y,X
    # If (1) or (2) are true, the function converts the stacks into the required format or returns error.
    
    
    if type(Im_Stack) is list and (type(Im_Stack[0]) is np.ndarray and Im_Stack[0].ndim == 2):
        return Im_Stack
    elif type(Im_Stack) is np.ndarray and Im_Stack.ndim == 2:
        Single_Image = Im_Stack
        Im_Stack = []
        Im_Stack.append(Single_Image)
        return Im_Stack
    elif type(Im_Stack) is np.ndarray and Im_Stack.ndim == 3:
        #Read Im_Stack as numpy.ndarray of 3 dimensions. Assuming stacks are along axis = 0
        Im_Stack = [Im_Stack[iS] for iS in range(Im_Stack.shape[0])]
        return Im_Stack
    else:
        error_message("\nSomething is wrong. Input to function could not be recognized as list of stacks. Returning nothing.")
        return []

def toArray(Im_Stack):
    
    # Ensures that a stack of images is converted to a 2D array (if single image) or 3D array if multiple images.
    # Just using "np.asarray" will convert a "stack" of a single image to 1xMxN array.
    if type(Im_Stack) == np.ndarray:
        if Im_Stack.ndim == 3 or Im_Stack.ndim == 2:
            return Im_Stack
        else:
            error_message("\nERROR: Too many or too few dimensions to interpret as image stack or image.")
            return
    elif type(Im_Stack) is list and (type(Im_Stack[0]) is np.ndarray and Im_Stack[0].ndim == 2):
        if len(Im_Stack) == 1:
            return Im_Stack[0]
        else:
            return np.asarray(Im_Stack)
    else:    
        error_message("\nERROR: Type not recognized.")
        return


def to16bit(Im_Stack, auto_adjust = False, method = None, norm_type = "global"):
    
    
    # Type casting to 16 bit with multiple options:
    # auto_adjust: If False, checks if input is already 16 bit and returns quickly without scaling or clamping the values over 0,2**16-1
    # method: Choose "normalize" if data should be scaled to full range (0,2**16-1) or "clamp" to clamp within that range
    
    bit_type = 2**16 - 1
    Im_Stack = toStack(Im_Stack)

    if (type(Im_Stack[0][0,0]) is np.uint16) and (auto_adjust is False):
        return Im_Stack

    
    if method == None or method == "clamp":
        Im_Stack = clamp(Im_Stack, limit_low = 0, limit_high = bit_type)
    elif method == "normalize":
        Im_Stack = normalize(Im_Stack, norm_type = norm_type, amax = bit_type)
    else:
        error_message("ERROR: argument 'norm_type' not recognized.")
        return []
    
    Im_Stack = np.asarray(Im_Stack).astype(np.uint16)
    Im_Stack = toStack(Im_Stack)
    
    return Im_Stack

def tofloat32(Im_Stack):
    
    
    # Type casting to 32 bit float:
    
    Im_Stack = toStack(Im_Stack)

    if (type(Im_Stack[0][0,0]) is np.float32):
        return Im_Stack
        
    Im_Stack = np.asarray(Im_Stack).astype(np.float32)
    Im_Stack = toStack(Im_Stack)
    
    return Im_Stack

def to8bit(Im_Stack, auto_adjust = False, method = None, norm_type = "global"):
    
    
    # Type casting to 8 bit with multiple options:
    # auto_adjust: If False, checks if input is already 8 bit and returns quickly without scaling or clamping the values over 0,2**8-1
    # method: Choose "normalize" if data should be scaled to full range (0,2**8-1) or "clamp" to clamp within that range
    
    bit_type = 2**8 - 1
    Im_Stack = toStack(Im_Stack)

    if (type(Im_Stack[0][0,0]) is np.uint8) and (auto_adjust is False):
        return Im_Stack

    
    if method == None or method == "clamp":
        Im_Stack = clamp(Im_Stack, limit_low = 0, limit_high = bit_type)
    elif method == "normalize":
        Im_Stack = normalize(Im_Stack, norm_type = norm_type, amax = bit_type)
    else:
        error_message("ERROR: argument 'norm_type' not recognized.")
        return []
    
    Im_Stack = np.asarray(Im_Stack).astype(np.uint8)
    Im_Stack = toStack(Im_Stack)
    
    return Im_Stack











def normalize(Im_Stack, norm_type = "global", amin = 0, amax = 2**16 - 1):

    #Normalization is done as: amin + (amax-amin)*(I-I_min)/(I_max-I_min)
    #norm_type: choose "global" to normalize using min/max values over entire stack else choose "local"
    
    
    Im_Stack = toStack(Im_Stack)
    Im_Stack = np.asarray(Im_Stack).astype(float)
    amin = float(amin)
    amax = float(amax)
    
    if norm_type == "local":
        alow = Im_Stack.min(axis = (1,2))
        ahigh = Im_Stack.max(axis = (1,2))
        Im_Stack = [np.nan_to_num(amin + (amax-amin)*(Im_Stack[iS] - alow[iS])/(ahigh[iS] - alow[iS])) for iS in range(Im_Stack.shape[0])]
    
    elif norm_type == "global":
        alow = Im_Stack.min()
        ahigh = Im_Stack.max()
        Im_Stack = [np.nan_to_num(amin + (amax-amin)*(Im_Stack[iS] - alow)/(ahigh - alow)) for iS in range(Im_Stack.shape[0])]
    else:
        error_message("ERROR: argument 'norm_type' not recognized.")
        return []

    return toStack(Im_Stack)

        
def clamp(Im_Stack, limit_low = None, limit_high = None):
    
    # Clamping: If values is lower than limit_low, replace it with limit_low, same for higher limit
    Im_Stack = np.copy(Im_Stack)
    
    Im_Stack = np.clip(Im_Stack, a_min = limit_low, a_max = limit_high)
    
    return toStack(Im_Stack)
        

def threshold(Im_Stack, a_th, a0 = None, a1 = None):
    
    # Input a threshold value "a_th". For values in the image stack lower than a_th, replace with a0 else with a1
    Im_Stack = np.copy(np.asarray(Im_Stack))
    
    idx_low = np.where(Im_Stack < a_th)
    idx_high = np.where(Im_Stack >= a_th)
    
    Im_Stack[idx_high] = a1
    Im_Stack[idx_low] = a0
    
    return toStack(Im_Stack)        


def modified_autocontrast(Im_Stack, s = 0.01, plot_flag = False):
    
    # return: alow, ahigh values to clamp data
    # s: quantile of image data to saturate. E.g. s = 0.01 means saturate the lowest 1% and highest 1% pixels
    message(BORDER)
    message("\nPerforming modified auto-contrast adjustment...")
    
    data_type  = np.asarray(Im_Stack).dtype
    
    S = np.copy(Im_Stack).astype(np.float32)
    
    
    if type(s) == tuple and len(s) == 2:
        slow, shigh = s
    else:
        slow = s
        shigh = s

    h, bins = np.histogram(S, bins = 500)
    c = np.cumsum(h)
    c_norm = c/np.max(c)
    
    ibin_low = np.argmin(np.abs(c_norm - slow))
    ibin_high = np.argmin(np.abs(c_norm - 1 + shigh))
    
    alow = bins[ibin_low]
    ahigh = bins[ibin_high]
    
    return alow, ahigh
    

def calc_EnsembleStack(Im_Stack, FPE, ret_type = INT16, ensemble_type = 'mean'):
    
    # Explanation: If you have a stack of 2000 frames of 20 identical events (100 frames in each event - FPE), and you need to reduce this data:
    # Choose 'mean' to get a mean stack with 100 frames. Corresponding frames in each event will be averaged.
    # Similarly, choose 'median' or 'stdev' for images containing these statistics.
    
    Im_Stack = toStack(Im_Stack)
    
    numFullEvents = int(len(Im_Stack)//FPE)
    message(BORDER)
    message("\nCalculating mean stack from %i events with %i frames per event"%(numFullEvents, FPE))

    t0 = time.time()

    # Numpy FTW! First slice the stack to eliminate incomplete events
    Im_Stack = np.asarray(Im_Stack[:numFullEvents*FPE])
    # Then slice the stack using start:stop:step as ii::FPE. Start at 'ii'th frame, then step FPE frames to next identical frame
    Ensemble_Stack = np.asarray([Im_Stack[ii::FPE,:,:] for ii in range(FPE)])
    # Now you get a 4D array with dim0 as time (length = FPE), dim1 as event (length = numFullEvents), dim2, dim3 as Y, X
    # So take the mean along events axis (axis 1)

    if ensemble_type == 'mean':
        Im_Stack = np.mean(Ensemble_Stack, axis = 1)
    elif ensemble_type == 'median':
        Im_Stack = np.median(Ensemble_Stack, axis = 1)
    elif ensemble_type == 'stdev':
        Im_Stack = np.std(Ensemble_Stack, axis = 1)
    else:
        error_message("ERROR: argument 'ensemble_type' not recognized.!")
        return []
    
    Im_Stack = toStack(Im_Stack)
    
    # If ret_type is INT16, convert to 16 bit, else return FLOAT32 values.
    if ret_type == INT16:
        Im_Stack = to16bit(Im_Stack)
    elif ret_type == FLOAT32:
        Im_Stack = Im_Stack
    # Done. This is done 10 times faster than the best for loop I could come up with.
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))

    return Im_Stack

    

def get_MovingStack(Im_Stack, cutoff = (0.2,0.7), ROIY = np.array([0,0]), ROIX = np.array([0,0]), LogFileName = ''): 
    
    # cutoff: Transience in frame is highest at 1.0, lowest at 0.0
    # ROI: Choose a region of interest (ROI) if known apriori
    # LogFileName: Pass a path to save plot.
    
    # Use this function if you want to estimate static frames in your stack.
    # This function takes Im_Stack, does a template_match on each image w.r.t. to the first frame,
    # then returns an array called "Trace" that denotes the "mismatch" on a scale of 0 - 1.
    # Using Trace, we slice the Im_Stack to get those frames where the mismatch is greater than 'cutoff'
    # These are effectively the frames where objects are in motion, the rest are "static background"

    message(BORDER)
    message("\nSearching for images in which object is constantly in motion...")
    Im_Stack = toStack(Im_Stack)

    t0 = time.time()
    
    Im_Stack = np.asarray(Im_Stack)
    Trace = get_MotionTrace(Im_Stack, ROIY, ROIX, LogFileName = LogFileName)

    if type(cutoff) is tuple:
        cutofflow, cutoffhigh = cutoff
        cutoffval_low = cutofflow*Trace.max()
        cutoffval_high = cutoffhigh*Trace.max()
        idx_array = np.where((Trace < cutoffval_high) & (Trace > cutoffval_low))
        idx_array = idx_array[0]
    else:
        error_message("Cut-off argument was not provided correctly... Returning nothing")
        return toStack(Im_Stack)
        
    Moving_Stack = Im_Stack[idx_array,:,:]
    Moving_Stack = [Moving_Stack[iS] for iS in range(Moving_Stack.shape[0])]
    
    t1 = time.time()
    message("\tFound %i out of %i."%(idx_array.shape[0],Im_Stack.shape[0]))
    message("\tDone in %f seconds."%(t1-t0))
    return Moving_Stack

##########################################################################

# MULTICHANNEL IMAGES

#########################################################################


def makeMultiChannel(a):
    
    # Pass a list of up to three image stacks to make a multichannel image stack and return as np.array (Z, Y, X, 3)
    # E.g. if a, b, c are three stacks, each (LxMxN), then pass as [a, b, c] python list
    a = np.copy(np.asarray(a))

    if a.ndim not in range(3,5):
        error_message("ERROR: Number of dimensions invalid for multichannel image stack")
        return
    elif a.ndim == 4 or a.ndim == 3:
        if a.shape[0] > 3:
            error_message("ERROR: Found more than 3 channels.")
            return
        elif a.shape[0] == 1:
            error_message("ERROR: Found only 1 channel. Can't do anything.")
            return
        elif a.shape[0] == 2:
            a = makeMultiChannel([a[0],a[1],np.zeros(a[1].shape)]) # Just make one channel full of zeros
        else:
            a = np.moveaxis(a, 0, a.ndim-1)
    return a


def RGB_Image_toStack(RGB_Img):
    
    RGB_Img = np.copy(np.asarray(RGB_Img))
    if RGB_Img.ndim != 3:
        error_message("ERROR: Invalid number of dimensions for RGB image. Expected 3")
    elif RGB_Img.shape[2] != 3:
        error_message("ERROR: Invalid number of channels found. Expected 3.")
    else:
        Im_Stack = [RGB_Img[:,:,ii] for ii in range(3)]
    
    return Im_Stack


#######################################################################
    
# FUNCTIONS FOR PARALLELIZATION USING MULTIPROCESSING
    
#######################################################################

# Modify this function as necessary. ListLen is the number of items that are distributed across the number of cores.

def Opt_Procs(ListLen):

    if ListLen == 1:
        procs = 1
    elif ListLen <= 100:
        procs = min(2, multiprocessing.cpu_count())
    elif ListLen <= 500:
        procs = min(4, multiprocessing.cpu_count())
    elif ListLen <= 1000:
        procs = min(8, multiprocessing.cpu_count())
    elif ListLen <= 5000:
        procs = min(16,multiprocessing.cpu_count())
    else:
        procs = multiprocessing.cpu_count()
    
    return procs

def Parallelize(ListIn, f, procs = -1, **kwargs):
    
    # This function packages the "starmap" function in multiprocessing for Python 3.3+ to allow multiple iterable inputs for the parallelized function.
    # ListIn: any python list such that each item in the list is a tuple of non-keyworded arguments passable to the function 'f' to be parallelized.
    # f: function to be parallelized. Signature must not contain any other non-keyworded arguments other than those passed as iterables.
    # Example:
    # def multiply(x, y, factor = 1.0):
    #   return factor*x*y
    # X = np.linspace(0,1,1000)
    # Y = np.linspace(1,2,1000)
    # XY = [ (x, Y[i]) for i, x in enumerate(X)] # List of tuples
    # Z = IP.Parallelize_MultiIn(XY, multiply, factor = 3.0, procs = 8)
    # Create as many positional arguments as required, but remember all must be packed into a list of tuples.
    
    if type(ListIn[0]) != tuple:
        ListIn = [(ListIn[i],) for i in range(len(ListIn))]
    
    reduced_argfunc = functools.partial(f, **kwargs)
    
    if procs == -1:
        procs = Opt_Procs(len(ListIn))
        

    if procs == 1:
        OutList = [reduced_argfunc(*ListIn[iS]) for iS in range(len(ListIn))]
    else:
        p = Pool(processes = procs)
        OutList = p.starmap(reduced_argfunc, ListIn)
        p.close()
        p.join()
    
    return OutList



def Z_Parallelize(Im_Stack, f, procs = -1, **kwargs):

    # This function parallizes over pixels in an image stack. E.g. to apply a function along "Z" in a stack I(Z,Y,X),
    # the job is distributed over Y,X.
    # Transform image stack into list of pixel-time arrays
    Pixel_List = toPixelList_fromStack(Im_Stack)
    
    # Parallel Processing
    Pixel_List_result = Parallelize(Pixel_List, f, procs = procs, **kwargs)    
    
    # Re-assemble image stack
    Im_Stack_result = toStack_fromPixelList(Pixel_List_result, Im_Stack[0].shape)
    
    return Im_Stack_result


def toPixelList_fromStack(Im_Stack):
    
    # Transform image stack into list of pixel-time arrays
    Flat_Stack = np.asarray([Im_Stack[iS].reshape(Im_Stack[iS].size) for iS in range(len(Im_Stack))])
    Pixel_List = [Flat_Stack[:,iP] for iP in range(Flat_Stack.shape[1])]
    return Pixel_List


def toStack_fromPixelList(Pixel_List, orig_shape):
    
    # Re-assemble image stack
    Pixel_List = np.asarray(Pixel_List)
    Flat_Stack = [Pixel_List[:,iS] for iS in range(Pixel_List.shape[1])]
    Im_Stack = [Flat_Stack[iS].reshape(orig_shape) for iS in range(len(Flat_Stack))]
    return Im_Stack


#######################################################################
    
# FUNCTIONS FOR GEOMETRIC TRANSFORMATIONS: ROTATE, ETC.
    
#######################################################################

def rotate_CCW_aboutCenter(Im_Stack, angle):

    # 'nuff said: rotate an image by 'angle' about the center pixel, counterclockwise.
    message(BORDER)
    message("\nRotating image by %f degrees CCW..."%angle)
    t0 = time.time()
    Im_Stack = toStack(np.copy(Im_Stack))
    rows, cols = Im_Stack[0].shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle,1)
    Im_Stack = [cv2.warpAffine(Im_Stack[iS],M,(cols,rows)) for iS in range(len(Im_Stack))]
    
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack


def rotate_CCW_aboutXY(Im_Stack, angle, YX = (0,0)):

    # Rotate the stack of images by 'angle' about the pixel input as YX, counterclockwise.
    message(BORDER)
    message("\nRotating image by %f degrees CCW..."%angle)
    t0 = time.time()
    Im_Stack = toStack(Im_Stack)
    rows, cols = Im_Stack[0].shape
    Y, X = YX
    M = cv2.getRotationMatrix2D((X,Y), angle,1)
    Im_Stack = [cv2.warpAffine(Im_Stack[iS],M,(cols,rows)) for iS in range(len(Im_Stack))]
    
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack


def crop(Im_Stack, X = None, Y = None):
    
    # Crop an image stack. X and Y input as X = [x1,x2], Y = [y1,y2], where x1, x2, y1, y2 are pixels.
    message(BORDER)
    message("\nCropping images...")
    t0 = time.time()
    Im_Stack = toStack(Im_Stack)

    X = [0,Im_Stack[0].shape[1]] if X == None else X
    Y = [0,Im_Stack[0].shape[0]] if Y == None else Y
    
    Im_Stack = [Img[Y[0]:Y[1],X[0]:X[1]] for Img in Im_Stack]
    
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack

def translate(Im_Stack, YX = [0,0], procs = -1, ret_type = None):
    
    # Translate images in the stack with a YX vector (list or np array) of size Zx2 where Z is the length of the stack.
    # Or just pass [y,x] to translate all images in the stack by same amount.
    message("\nPerforming image translation...")
    t0 = time.time()
    
    Im_Stack = toStack(Im_Stack)

    if np.shape(YX) == (2,):
        YX = [np.asarray(YX) for iS in range(len(Im_Stack))]
    elif np.shape(YX) == (len(Im_Stack),2):
        YX = [np.asarray(YX[iS]) for iS in range(len(Im_Stack))]
    else:
        error_message("\nERROR: Transformation coordinates not provided in specified format. Shape must be (nImages,2)")
        return Im_Stack        
    
    ListIn = [(Im_Stack[ii], YX[ii]) for ii in range(np.shape(YX)[0])]
    
    Im_Stack = Parallelize(ListIn, _translate, procs = procs)
    
    if ret_type == INT16:    
        Im_Stack = to16bit(Im_Stack)
    
    Im_Stack = toStack(Im_Stack)
    
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack
    

def _translate(Img, YX, **kwargs):
    
    rows, cols = Img.shape
    M = np.float32([[1,0,YX[1]], [0,1,YX[0]]])
    
    return cv2.warpAffine(Img, M, (cols,rows))

    
    
    
        


#########################################################################
    
# FUNCTIONS FOR IMAGE CALCULATION, e.g. DIVIDE, SUBTRACT, ETC.
    
#########################################################################

def _AlphaBlend_Images(A, B, alpha = 0.5, **kwargs):
    # resultant = alpha*A + (1-alpha)*B
    alpha = np.float32(alpha)
    
    Float_Image = np.nan_to_num( alpha * A + (1.0 - alpha) * B )
    return Float_Image

def _AlphaBeta_AddImages(A, B, alpha  = 0.5, beta  = 0.5, **kwargs):
    # resultant = alpha*A + beta*B
    alpha = np.float32(alpha)
    beta = np.float32(beta)
    
    Float_Image = np.nan_to_num( alpha * A + beta * B)
    return Float_Image

def _Subtract_Images(A, B, **kwargs):
    # resultant = A - B
    
    Float_Image = np.nan_to_num(A - B)
    return Float_Image

def _Divide_Images(A, B, **kwargs):
    # resultant = A/B or 0 if B == 0
    
    Float_Image = np.nan_to_num(A/B)
    return Float_Image    

    

def Stack_Arithmetic(A, B, op = 'alphablending', alpha = -1, beta = -1 , procs = -1, ret_type = None):
    
    # Perform arithmetic on really large stacks. Use this function if numpy operations run slow on your computer.
    
    funcnames = {'alphablending' : _AlphaBlend_Images,
                 'A-B'           : _Subtract_Images,
                 'A/B'           : _Divide_Images,
                 'alphabeta'     : _AlphaBeta_AddImages}

    t0 = time.time()
    message(BORDER)
    message("\nPerforming image arithmetic of type %s..."%(op))

    if op not in funcnames.keys():
        error_message("\tERROR: Arithmetic operation not found...")
        return A

    A = toStack(A)
    B = toStack(B)
    # convert to float32 for better accuracy / prevent false zeroes, etc.
    A = [np.float32(A[iS]) for iS in range(len(A))]
    B = [np.float32(B[iS]) for iS in range(len(B))]
    
    if procs == -1:
        procs = multiprocessing.cpu_count()
    
    if len(A) != len(B):
        if len(A) == 1:
            A = [A[0] for iS in range(len(B))]
        elif len(B) == 1:
            B = [B[0] for iS in range(len(A))]
        else:
            error_message("ERROR: Image Stack lengths are not equal. Image Arithmetic failed.")
            return A
    
    ListIn = [(A[ii], B[ii]) for ii in range(np.shape(A)[0])]

    Result_Stack = Parallelize(ListIn, funcnames[op], procs = procs, alpha = alpha, beta = beta)
    

    # If ret_type is INT16, convert to 16 bit, else return float32 values.
    if ret_type == INT16:
        Result_Stack = to16bit(Result_Stack, method = "normalize", norm_type = "global")
    
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Result_Stack
    



######################################################################################

# FILTERS: MEDIAN, GAUSSIAN, Z-MEDIAN, Z-GAUSSIAN, etc.

######################################################################################
    
# Highly parallelized filters. Use these to run on large image stacks.

    

def get_sigma(kern_size):
    return 0.3*(0.5*(kern_size - 1) - 1) + 0.8

def XY_gaussianBlur(Im_Stack, X_kern_size = 3, Y_kern_size = 3, procs = -1, ret_type = None, X_sigma = None, Y_sigma = None):

    # Performs blur on each image separately, parallelized over all images.
    message(BORDER)
    message("\nPerforming XY Gaussian Blur...")
    t0 = time.time()
    
    X_sigma = get_sigma(X_kern_size) if not X_sigma else X_sigma
    Y_sigma = get_sigma(Y_kern_size) if not Y_sigma else Y_sigma
    
    Im_Stack = toStack(Im_Stack)
    Im_Stack = Parallelize(Im_Stack, scipyfilters.gaussian_filter, procs = procs,
                           sigma = (X_sigma, Y_sigma), order = 0)
    
    if ret_type == INT16:
        Im_Stack = to16bit(Im_Stack)
        

    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack

def Z_gaussianBlur(Im_Stack, kern_size = 3, procs = -1, ret_type = None):
    
    # Performs blur over the first axis (Z). Parallelized over all pixels in the image.
    message(BORDER)
    message("\nPerforming Z-directional Gaussian Blur...")
    t0 = time.time()
    Im_Stack = toStack(Im_Stack)
    
    Im_Stack = Z_Parallelize(Im_Stack, scipyfilters.gaussian_filter, procs = procs,  
                                    sigma = get_sigma(kern_size), order = 0)

    if ret_type == INT16:
        Im_Stack = to16bit(Im_Stack)
    
    
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack


def XYZ_medianBlur(Im_Stack,X_kern_size = 3, Y_kern_size = 3, Z_kern_size = 3, ret_type = None):
    
    message(BORDER)
    message("\nPerforming XYZ median filter...")
    Im_Stack = toStack(Im_Stack)

    # The scipy.ndimage median filter is faster algorithm compared to scipy.signal, but does not exploit parallel processing.
    # Hence, use parallelized versions Z_medianBlur and XY_medianBlur which perform XY and Z filters separately.
    # This function is provided because median filter is non-linear, so XYZ filter is not same as XY followed by Z.
    
    kern_size_tuple = (Z_kern_size, Y_kern_size, X_kern_size)
    t0 = time.time()
    Im_Stack = scipyfilters.median_filter(np.asarray(Im_Stack), kern_size_tuple)
    Im_Stack = [Im_Stack[iS] for iS in range(Im_Stack.shape[0])]
    
    if ret_type == INT16:
        Im_Stack = to16bit(Im_Stack)

    t1 = time.time()
    
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack


def XY_medianBlur(Im_Stack, X_kern_size = 3, Y_kern_size = 3, procs = -1, ret_type = None):

    # Performs blur on each image separately, parallelized over all images.
    
    message(BORDER)
    message("\nPerforming XY median filter...")
    t0 = time.time()
    
    Im_Stack = toStack(Im_Stack)
    Im_Stack = Parallelize(Im_Stack, scipyfilters.median_filter, procs = procs, size = (Y_kern_size, X_kern_size))
    
    if ret_type == INT16:
        Im_Stack = to16bit(Im_Stack)
    
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack

def Z_medianBlur(Im_Stack, Z_kern_size = 3, procs = -1, ret_type = None):

    # Performs blur over the first axis (Z). Parallelized over all pixels in the image.
    
    message(BORDER)
    message("\nPerforming Z-directional median blur...")
    t0 = time.time()
    
    Im_Stack = toStack(Im_Stack)
    Im_Stack = Z_Parallelize(Im_Stack, scipyfilters.median_filter, procs = procs,  size = Z_kern_size)

    if ret_type == INT16:
        Im_Stack = to16bit(Im_Stack)
    
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack


def _UnsharpMask_Image(A, MaskWeight = 0.6, Blur_size = 11, divide_flag = False, **kwargs):
    
    
    B = np.float32(scipyfilters.gaussian_filter(A, sigma = 0.3*(0.5*(Blur_size - 1) - 1) + 0.8, order = 0))
    A = np.float32(A)

    return np.nan_to_num(A / B) if divide_flag else np.nan_to_num(A - MaskWeight*B)


def UnsharpMask(Im_Stack, Blur_size = 11, MaskWeight = 0.6, divide_flag = False, procs = -1, ret_type = None):

    # Unsharp Mask with two options: division or subtraction.
    # divide_flag = False: A - Blurred(A)*MaskWeight
    # divide_flag = True: A/Blurred(A)
    
    message(BORDER)
    message("\nPerforming Unsharp Mask ...")
    t0 = time.time()
    Im_Stack = toStack(Im_Stack)
    
    
    Im_Stack = Parallelize(Im_Stack, _UnsharpMask_Image, procs = procs, divide_flag = divide_flag, Blur_size = Blur_size, MaskWeight = MaskWeight)
    
    if ret_type == INT16:
        Im_Stack = to16bit(Im_Stack, method = "normalize")
    
    
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))

    return Im_Stack



#################################################################################################

# FUNCTIONS FOR EDGE DETECTION

#################################################################################################

def calc_sobel(Im_Stack, procs = -1, ret_type = None, ret_format = list):

    message(BORDER)
    message("\nCalculating sobel transform...")
    t0 = time.time()
    Im_Stack = toStack(Im_Stack)

    Im_Stack = Parallelize(Im_Stack, skimagefilters.sobel, procs = procs)

    if ret_type == INT16:
        Im_Stack = to16bit(Im_Stack)
     

    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack if ret_format == list else toArray(Im_Stack)


    
def canny_edge(Im_Stack, sigma = 1.0, procs = -1, ret_type = None):

    message(BORDER)
    message("\nPerforming Canny edge detection...")
    t0 = time.time()
    Im_Stack = toStack(Im_Stack)

    Im_Stack = Parallelize(Im_Stack, canny, sigma = sigma, procs = procs)

    if ret_type == INT16:
        Im_Stack = to16bit(Im_Stack)

    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return Im_Stack
    

if __name__ == '__main__':
    print("Nothing to do here.")
