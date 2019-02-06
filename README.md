# ImageStackPy
Modules to view and process large image stacks. Fast, parallelized filters and an object tracking algorithm.

If you find yourself post-processing stacks of greyscale images such as those arising from high-speed imaging experiments, or video recordings and find that writing loops or list comprehensions over image processing functions in the opencv or skimage libraries does not utilized your processor efficiently, you might find my repository useful. Also, I offer easy-to-use functions to generate a motion trajectory of an object moving in the video sequence.
ImageProcessing.py file includes the following important functions, all of them are parallelized using multiprocessing library (somewhat similar to how you parallelized the template match algorithm)
 
The "PyImageProcessing.py" file contains:

1. XY / Z median, Gaussian and mean filters parallelized using multiprocessing.
2. Point operations: brightness adjustment / normalization, binarization / thresholding, unsharp mask filter.
3. Image calculation: addition, subtraction, division, alphablending - when numpy slows you down.
4. Read / write image stacks - currently only for tif / tiff format 16 bit image stacks.
5. Affine transforms - Image rotation / translation / cropping
6. Edge detection - Sobel, Canny

ObjectTracking.py file includes an object tracking algorithm using normalized cross-correlation (template matching). Input limits for a bounding box that contains the object in the first frame and find it's motion trajectory through the sequence of images. Output as an XY vector or draw a bounding box over the moving object through the stack of images.

Define an image stack as a python list of 2D numpy arrays with identical shape - I(Z,Y,X). The 'Z axis' is the python list.

Example:
import ImageProcessing as IP
Im_Stack = IP.get_stack(userfilepath = "Enter path")
Im_Stack = IP.XY_gaussianBlur(Im_Stack, X_kern_size = 3, Y_kern_size = 3)
