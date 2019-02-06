#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:33:53 2018

@author: atekawade

Image visualization: Requires image arrays (2D) or volume arrays (3D) as np arrays.
To visualize multiple volumes at a time and choose to toggle between them, pass them as a list:
E.g. if a and b are two np arrays with ndim = 3 each, pass them as: [a,b]

"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, SpanSelector
from matplotlib.figure import Figure
from matplotlib import patches

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

def get_copy(list_in):
    array = np.copy(list_in)
    return [array[ii] for ii in range(np.shape(array)[0])]


def viewer(ims_s, names = None, cmap = 'Greys'):
    
    # View a greyscale single channel image or image stack.
    # ims_s: Either of the following:
    # 1. Image stack I(Z,Y, X) or 3D volume as numpy array (ndim = 3): This will display the stack / 3D volume
    # 2. List of 3D volumes N x I(Z,Y,X) - will display one 3D volume at a time with radio button to pick the right volume
    # 3. Just an image or 2D numpy array.

    # names: pass a list of names of the stacks passed to see as labels for buttons.
    
    if np.ndim(ims_s) == 2:
        ims_s = IP.toStack(ims_s)
    
    if np.ndim(ims_s) == 3:
        ims_s = [ims_s]
    

    # Use a COPY and Fix NaNs. Never forget to use copies. Numpy is a leaky API.
    ims_s = get_copy(ims_s)
    for im in ims_s: im[np.isnan(im) == True] = 0

    if not names: names = ["#" + str(ii+1) for ii in range(len(ims_s))]    
        
    numFrames = ims_s[0].shape[0]
    rax_textlen = len(max(names, key = len))*0.02

    
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25, left = rax_textlen + 0.08)
    ax.axis('off')
    im = plt.imshow(ims_s[0][0], cmap = cmap)
    
    ax_num = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor = 'red') # Frame number slider
    ax_max = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor = 'blue') # Max value slider
    ax_min = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor = 'green') # Min value slider
    rax = plt.axes([0.025, 0.5, rax_textlen, 0.15], facecolor = 'lightgoldenrodyellow') #Radio buttons to choose between processed and original image
    
    
    img_slider = Slider(ax_num, 'FRAME', 1, numFrames, valinit = 1, valstep = 1, valfmt = '%i')
    max_slider = Slider(ax_max, 'CLAMP_MAX', 0.00, 1.00, valinit = 1.00)
    min_slider = Slider(ax_min, 'CLAMP_MIN', 0.00, 1.00, valinit = 0.00)
    radio = RadioButtons(rax, names, active = 0)
    
    # Calculate transform of intensity values from min, max to 0, 1.
    minv = [np.min(np.float32(ims_s[ii])) for ii in range(len(ims_s))]
    maxv = [np.max(np.float32(ims_s[ii])) for ii in range(len(ims_s))]
    
    def intensity_map(xmin, xmax, minv, maxv):
        return xmin*(maxv - minv) + minv, xmax*(maxv - minv) + minv
    
    def update(blah): 
        
        num = img_slider.val
        maxval = max(max_slider.val, min_slider.val)
        minval = min(max_slider.val, min_slider.val)
        idx = names.index(radio.value_selected)
        im.set_data(ims_s[idx][int(num)-1])
    
        im.set_clim(intensity_map(minval, maxval, minv[idx], maxv[idx]))
                    

        fig.canvas.draw_idle()
        
        
    
    img_slider.on_changed(update)
    max_slider.on_changed(update)
    min_slider.on_changed(update)
    radio.on_clicked(update)

    
    ax_num._slider = img_slider

    ax_max._slider = max_slider
    ax_min._slider = min_slider
    rax._button = radio
    
    return


def plot_profile(ims_s, names = None):
    
    # View a greyscale single channel image or image stack and plot profiles along X and / or Y axis.
    # ims_s: Either of the following:
    # 1. Image stack I(Z,Y, X) or 3D volume as numpy array (ndim = 3): This will display the stack / 3D volume
    # 2. List of 3D volumes N x I(Z,Y,X) - will display one 3D volume at a time with radio button to pick the right volume
    # 3. Just an image or 2D numpy array.

    # names: pass a list of names of the stacks passed to see as labels for buttons.
    
    if np.ndim(ims_s) == 2:
        ims_s = IP.toStack(ims_s)
    
    if np.ndim(ims_s) == 3:
        ims_s = [ims_s]
    

    # Use a COPY and Fix NaNs. Never forget to use copies. Numpy is a leaky API.
    ims_s = get_copy(ims_s)
    for im in ims_s: im[np.isnan(im) == True] = 0

    
    
    if not names: names = ["#" + str(ii+1) for ii in range(len(ims_s))]    

    numFrames = ims_s[0].shape[0]
    X_size = ims_s[0].shape[2]
    Y_size = ims_s[0].shape[1]
    
    cols = ['red', 'green', 'blue', 'orange', 'pink']

    # Calculate transform of intensity values from min, max to 0, 1.
    minv = [np.min(np.float32(ims_s[ii])) for ii in range(len(ims_s))]
    maxv = [np.max(np.float32(ims_s[ii])) for ii in range(len(ims_s))]
    
    fig, ax = plt.subplots(1,3)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    plt.subplots_adjust(bottom=0.4, left = 0.08)
    
    #ax[0].axis('off')
    im = ax[0].imshow(ims_s[0][0], cmap =  'gray')
    
    X_prof = [0]*len(names)
    Y_prof = [0]*len(names)
    for ii in range(len(names)):
        X_prof[ii] = ax[1].plot(ims_s[ii][0,Y_size//2,:], color = cols[ii], label = names[ii])
        ax[1].set_title('X profile')
        #ax[1].set_yticks([])

        #ax[1].legend()

    for ii in range(len(names)):
        Y_prof[ii] = ax[2].plot(ims_s[ii][0,:,X_size//2], color = cols[ii], label = names[ii])
        ax[2].set_title('Y profile')
        #ax[2].set_yticks([])
    
    ax[1].legend(loc = 'topright')

    
    slider_Ypos = np.arange(5)*0.04 + 0.01
    slider_Xpos = 0.2
    ax_num = plt.axes([slider_Xpos, slider_Ypos[0], 0.65, 0.02], facecolor = 'red') # Frame number slider
    ax_max = plt.axes([slider_Xpos, slider_Ypos[1], 0.65, 0.02], facecolor = 'blue') # Max value slider
    ax_min = plt.axes([slider_Xpos, slider_Ypos[2], 0.65, 0.02], facecolor = 'green') # Min value slider
    ax_Y = plt.axes([slider_Xpos,   slider_Ypos[3], 0.65, 0.02], facecolor = 'green') # Y pixel slider
    ax_X = plt.axes([slider_Xpos,   slider_Ypos[4], 0.65, 0.02], facecolor = 'blue') # X pixel slider
    rax = plt.axes([0.025, slider_Ypos[4], 0.05, 0.05*len(names)], facecolor = 'lightgoldenrodyellow') #Radio buttons to choose between processed and original image
    axupdate = plt.axes([slider_Xpos, slider_Ypos[4] + 0.04, 0.1, 0.05])
    
    img_slider = Slider(ax_num, 'FRAME', 1, numFrames, valinit = 1, valstep = 1, valfmt = '%i')
    max_slider = Slider(ax_max, 'CLAMP_MAX', 0.00, 1.00, valinit = 1.00)
    min_slider = Slider(ax_min, 'CLAMP_MIN', 0.00, 1.00, valinit = 0.00)
    X_slider = Slider(ax_X, 'X coord.', 0, X_size, valinit = 0, valstep = 1, valfmt = '%i')
    Y_slider = Slider(ax_Y, 'Y coord.', 0, Y_size, valinit = 0, valstep = 1, valfmt = '%i')
    radio = RadioButtons(rax, names, active = 0)
    update_button = Button(axupdate, 'Update')
    
    def intensity_map(xmin, xmax, minv, maxv):
        return xmin*(maxv - minv) + minv, xmax*(maxv - minv) + minv
    
    def update(blah): 
        
        num = img_slider.val
        maxval = max(max_slider.val, min_slider.val)
        minval = min(max_slider.val, min_slider.val)
        idx = names.index(radio.value_selected)
        Y_pt = int(Y_slider.val)
        X_pt = int(X_slider.val)
        
        temp = np.copy(ims_s[idx][int(num)-1])
        temp[Y_pt,:] = maxv[idx]
        temp[:,X_pt] = maxv[idx]
        im.set_data(temp)
        
        
        
            
        im.set_clim(intensity_map(minval, maxval, minv[idx], maxv[idx]))
                    

        fig.canvas.draw_idle()
        fig.canvas.draw_idle()
        
        
    def update_prof(bleh):

        num = img_slider.val
        Y_pt = int(Y_slider.val)
        X_pt = int(X_slider.val)

        for ii in range(len(names)):
            X_prof[ii][0].set_ydata(ims_s[ii][int(num)-1,Y_pt,:])
    
        for ii in range(len(names)):
            Y_prof[ii][0].set_ydata(ims_s[ii][int(num)-1,:,X_pt])
            
        ax[1].set_ylim([ np.min(np.asarray(ims_s)[:,int(num)-1,Y_pt,:]), np.max(np.asarray(ims_s)[:,int(num)-1,Y_pt,:])    ])
        ax[2].set_ylim([ np.min(np.asarray(ims_s)[:,int(num)-1,:,X_pt]), np.max(np.asarray(ims_s)[:,int(num)-1,:,X_pt])    ])

        
    
    img_slider.on_changed(update)
    max_slider.on_changed(update)
    min_slider.on_changed(update)
    radio.on_clicked(update)
    update_button.on_clicked(update_prof)
    X_slider.on_changed(update)
    Y_slider.on_changed(update)
    
    
    ax_num._slider = img_slider

    ax_max._slider = max_slider
    ax_min._slider = min_slider
    ax_Y._slider = Y_slider
    ax_X._slider = X_slider
    rax._button = radio
    axupdate._button = update_button

    
    
    #plt.close()
    #return fig
    return




######################## 
    

def plot_histogram(ims_s, names = None, bins = None):
    
    # View a greyscale single channel image or image stack and view histogram.
    # ims_s: Either of the following:
    # 1. Image stack I(Z,Y, X) or 3D volume as numpy array (ndim = 3): This will display the stack / 3D volume
    # 2. List of 3D volumes N x I(Z,Y,X) - will display one 3D volume at a time with radio button to pick the right volume
    # 3. Just an image or 2D numpy array.

    # names: pass a list of names of the stacks passed to see as labels for buttons.
    
    if np.ndim(ims_s) == 2:
        ims_s = IP.toStack(ims_s)
    
    if np.ndim(ims_s) == 3:
        ims_s = [ims_s]
    

    # Use a COPY and Fix NaNs. Never forget to use copies. Numpy is a leaky API.
    ims_s = get_copy(ims_s)
    for im in ims_s: im[np.isnan(im) == True] = 0

    
    if not names: names = ["#" + str(ii+1) for ii in range(len(ims_s))]    

    if bins == None: bins = 100
    numFrames = ims_s[0].shape[0]
    X_size = ims_s[0].shape[2]
    Y_size = ims_s[0].shape[1]
    
    cols = ['red', 'green', 'blue', 'orange', 'pink']

    # Calculate transform of intensity values from min, max to 0, 1.
    minv = [np.min(np.float32(ims_s[ii])) for ii in range(len(ims_s))]
    maxv = [np.max(np.float32(ims_s[ii])) for ii in range(len(ims_s))]
    
    fig, ax = plt.subplots(1,2)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    plt.subplots_adjust(bottom=0.4, left = 0.08)
    
    #ax[0].axis('off')
    im = ax[0].imshow(ims_s[0][0])
    
    hists = [0]*len(names)
    
    for ii in range(len(names)):
        hists[ii] = ax[1].hist(ims_s[ii][0].ravel(), bins = bins, label = names[ii])
        ax[1].set_title('Intensity Histogram')
        #ax[1].set_yticks([])

        #ax[1].legend()
    
    ax[1].legend(loc = 'topright')

    
    slider_Ypos = np.arange(5)*0.04 + 0.01
    slider_Xpos = 0.2
    ax_num = plt.axes([slider_Xpos, slider_Ypos[0], 0.65, 0.02], facecolor = 'red') # Frame number slider
    ax_max = plt.axes([slider_Xpos, slider_Ypos[1], 0.65, 0.02], facecolor = 'blue') # Max value slider
    ax_min = plt.axes([slider_Xpos, slider_Ypos[2], 0.65, 0.02], facecolor = 'green') # Min value slider
    rax = plt.axes([0.025, slider_Ypos[4], 0.05, 0.05*len(names)], facecolor = 'lightgoldenrodyellow') #Radio buttons to choose between processed and original image
    axupdate = plt.axes([slider_Xpos, slider_Ypos[4] + 0.04, 0.1, 0.05])
    
    img_slider = Slider(ax_num, 'FRAME', 1, numFrames, valinit = 1, valstep = 1, valfmt = '%i')
    max_slider = Slider(ax_max, 'CLAMP_MAX', 0.00, 1.00, valinit = 1.00)
    min_slider = Slider(ax_min, 'CLAMP_MIN', 0.00, 1.00, valinit = 0.00)
    radio = RadioButtons(rax, names, active = 0)
    update_button = Button(axupdate, 'Update')
    
    def intensity_map(xmin, xmax, minv, maxv):
        return xmin*(maxv - minv) + minv, xmax*(maxv - minv) + minv
    
    def update(blah): 
        
        num = img_slider.val
        maxval = max(max_slider.val, min_slider.val)
        minval = min(max_slider.val, min_slider.val)
        idx = names.index(radio.value_selected)
        
        im.set_data(ims_s[idx][int(num)-1])
        
            
        im.set_clim(intensity_map(minval, maxval, minv[idx], maxv[idx]))
                    

        fig.canvas.draw_idle()
        
        
    def update_prof(bleh):

        num = img_slider.val

        #for ii in range(len(names)):
        #    hists[ii][0].set_data(ims_s[ii][int(num)-1].ravel(), bins = bins,  label = names[ii])
    
        ax[1].cla()
        for ii in range(len(names)):
            hists[ii] = ax[1].hist(ims_s[ii][int(num)-1].ravel(), bins = bins, label = names[ii])
            ax[1].set_title('Intensity Histogram')
            #ax[1].set_yticks([])
    
            #ax[1].legend()
        ax[1].legend(loc = 'topright')

        
    
    img_slider.on_changed(update)
    max_slider.on_changed(update)
    min_slider.on_changed(update)
    radio.on_clicked(update)
    update_button.on_clicked(update_prof)
    
    
    
    
    
    ax_num._slider = img_slider

    ax_max._slider = max_slider
    ax_min._slider = min_slider
    rax._button = radio
    axupdate._button = update_button

    
    #plt.close()
    #return fig
    return







