# ImageStackPy
### Modules to view and process large image stacks. Fast, parallelized filters and an object tracking algorithm.

Library for post-processing stacks of greyscale images such as those arising from high-speed imaging experiments, or video recordings and need to parallelize functions in opencv / skimage / numpy.

ImageProcessing.py file includes the following important functions, all of them are parallelized using multiprocessing library (somewhat similar to how you parallelized the template match algorithm)
 
The "PyImageProcessing.py" file contains:

1. 3D filters (blur, edge detection, unsharp mask, etc.) parallelized using multiprocessing for Python.
2. Point operations: brightness adjustment / normalization, binarization / thresholding.
3. Image calculation: addition, subtraction, division, alphablending - when numpy slows you down.
4. Read / write image stacks - tif / tiff format 16 bit image stacks.
5. Affine transforms - Image rotation / translation / cropping


ObjectTracking.py file includes an object tracking algorithm using normalized cross-correlation (template matching). Input limits for a bounding box that contains the object in the first frame and find it's motion trajectory through the sequence of images. Output as an XY vector or draw a bounding box over the moving object through the stack of images.

Img_Viewer.py contains functions to view image stacks in a slider window using matplotlib interactive widgets. You can also view the histogram, plot profile of pixel intensity across images, etc.

An image stack is defined  as a python list of 2D numpy arrays with identical shape - I(Z,Y,X). The 'Z axis' is the python list.

####Installation:

Use a dedicated python 3 environment,

pip install git+https://github.com/aniketkt/ImageStackPy.git#egg=ImageStackPy

Example:
from ImageStackPy import ImageProcessing as IP

import time

Im_Stack = IP.get_stack(userfilepath = "path/pathdir")

t0 = time.time()

Im_Stack1 = IP.XY_gaussianBlur(Im_Stack, X_kern_size = 3, Y_kern_size = 3)

print("Took %.2f secs"%(time.time() - t0))


from ImageStackPy import Img_Viewer as VIEW

Im_Stack1 = IP.toArray(Im_Stack1)

VIEW.viewer(Im_Stack1)

