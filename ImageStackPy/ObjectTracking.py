#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:56:50 2018

@author: atekawade
"""

import numpy as np
from skimage.feature import match_template
import time


from ImageStackPy import ImageProcessing as IP

DEBUG_MODE = False

# Suppress print input / output

def message(message_str):
    
    if DEBUG_MODE == True:
        print(message_str) 
    return

def error_message(message_str):
    
    print(message_str) 
    return




# Takes a template and image, locates the template within the image and returns the Y, X indices corresponding to the center of the location.
def locate(image, template = []):
    match_zone = match_template(image, template, pad_input = True)
    return np.unravel_index(np.argmax(match_zone), match_zone.shape)


def track_original(Im_Stack, ROIX = [0,0], ROIY = [0,0], procs = -1):
    
    # Using a template based on the first frame, tracks the location of the feature in the template in all subsequent frames.
    # The returned variable is a numpy array of shape [nFrames,2], giving the Y, X "displacement" of the feature in all frames.
    # ROIY & ROIX: Provide the Y and X pixel range to define the region of interest that will move and needs tracking.
    
    message(IP.BORDER)
    message("\nTracking location based on template at t = 0...")
    t0 = time.time()
    
    if not (any(ROIX) and any(ROIY)):
        return np.asarray([0,0])
    Im_Stack = IP.check_format(Im_Stack)
    
    template = IP.crop(IP.check_format(Im_Stack[0]), X = ROIX, Y = ROIY)[0]
    YX = np.asarray(IP.Parallelize(Im_Stack, locate, procs = procs, template = template))
    
    t1 = time.time()
    message("\tDone in %f seconds."%(t1-t0))
    return YX - YX[0,:]



def show_tracking(Im_Stack, YX, ROIX = [0,0], ROIY = [0,0], multichannel = False, fillBox = False, BoxBrightness = 0.2):
    
    # Draws a box around the region of interest and using the tracking vector 'YX', returns a stack showing the motion of the ROI / box.
    # YX: numpy array (Z,2) with displacement data for each frame. Frame #1 should be [0,0]
    # ROIY & ROIX: Provide the Y and X pixel range to define the region of interest that will move and needs tracking.
    # multichannel = True if you need the box in a different color, fillBox = False if you just need a bounding box, True if it should be colored.
    
    
    message(IP.BORDER)
    message("\nGenerating image sequence showing the tracked object...")
    t0 = time.time()
    
    W = int((ROIX[1] - ROIX[0]))
    H = int((ROIY[1] - ROIY[0]))
    
    CX_0 = int((ROIX[0] + ROIX[1])/2)
    CY_0 = int((ROIY[0] + ROIY[1])/2)    
    YX = YX + np.asarray([CY_0, CX_0])

    BOX = [draw_box(Im_Stack[iS], W = W , H = H, Center = YX[iS], filled = fillBox) for iS in range(len(Im_Stack))]
    
    BOX = IP.clamp(BOX, limit_low = np.min(Im_Stack), limit_high = np.max(Im_Stack))
    
    if multichannel == True:
        return IP.makeMultiChannel([Im_Stack, BOX])
    else:
        Im_Stack = IP.Stack_Arithmetic(Im_Stack, BOX, op = 'alphablending', alpha = 1 - BoxBrightness)
    
    t1 = time.time()
    message("\nFinished generating image sequence showing the tracked object, took %f seconds"%(t1-t0))
    
    return Im_Stack
    

def draw_box(Image, W = 0, H = 0, Center = np.array([0,0]), filled = False):
    
    Y, X = Center
    IH = Image.shape[0]
    IW = Image.shape[1]
    
    #Is Center within the Image Space?
    if not ((Y in range(IH)) and (X in range(IW))):
        error_message("ERROR: Box center is not within image space.")
        return
    
    centerpixel_right = IW - (X + 1)
    centerpixel_left = X
    centerpixel_top = Y
    centerpixel_bottom = IH - (Y + 1)
    
    
    box_right = min( int(W/2) , centerpixel_right )
    box_left = min( int(W/2), centerpixel_left )
    box_bottom = min( int(H/2) , centerpixel_bottom )
    box_top = min( int(H/2), centerpixel_top )
    
    BOX = (2**16 - 1)*np.ones((box_bottom + box_top + 1,box_right + box_left + 1))
    if filled == False: BOX[3:-3,3:-3] = 0
    
    pad_right = centerpixel_right - box_right
    pad_left = centerpixel_left - box_left
    pad_top = centerpixel_top - box_top
    pad_bottom = centerpixel_bottom - box_bottom
    
    BOX = np.pad(BOX,((pad_top,pad_bottom),(pad_left,pad_right)), 'constant', constant_values = 0)
    return BOX


    
 
 

    
    

    
    
    
