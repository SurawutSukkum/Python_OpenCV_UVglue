import cv2
from segment_anything import SamAutomaticMaskGenerator
import os
import torch
from segment_anything import sam_model_registry
import supervision as sv
from time import process_time
import gc
torch.cuda.memory_summary(device=None, abbreviated=False)
import torch.utils.checkpoint as checkpoint
from torch.utils.checkpoint import checkpoint_sequential
os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:512' #'garbage_collection_threshold:0.8,max_split_size_mb:512'
import numpy as np
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
segmentor = SelfiSegmentation()

# Constants for calibration
reference_width_mm = 10  # Width of the reference object in mm
reference_width_pixels = 100  # Width of the reference object in the image in pixels

HOME = os.getcwd()
IMAGE_PATH = 'UV.JPEG'
IMAGE_PATH = os.path.join(HOME, IMAGE_PATH)
print(IMAGE_PATH)

# Start the stopwatch / counter 
t1_start = process_time()
vid = cv2.VideoCapture(1)       

while True:
    t1_start = process_time()
    image = cv2.imread(IMAGE_PATH)
    blue, green, red = cv2.split(image)

    scale_percent = 100 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    print(image.shape) 

    x =int( width / 2)+300
    y = 1800
    w = 2000
    h = 1000
    
    #cropped_image = image[y:y+h, x:x+w]
    # Start the stopwatch / counter 
    
    cropped_image = image
    image_contour_thred1 = cropped_image.copy()
    image_contour_org1 = cropped_image.copy()
    gray = cv2.cvtColor(image_contour_thred1, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)  # Apply thresholding4
    #cv2.imshow('Contour detection using threshold', threshold)
    contours1, hierarchy1 = cv2.findContours(image=threshold, mode=cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)    
    min_area_threshold = 20000  # Minimum area threshold to filter out small contours

    # Calculate scale factor for converting pixels to mm
    scale_factor = reference_width_mm / reference_width_pixels

    c = 0 
    for contour in contours1:
     c = c+1
     print(len(contours1))
     area_pixels = cv2.contourArea(contour)
     # Convert the area from pixels to mm
     area_mm = area_pixels * scale_factor * scale_factor     
     if  area_pixels > min_area_threshold:
        print("Contour area:", area_pixels)
        print()
        print(c)
        image_contour_org1  = cv2.putText( image_contour_org1 , "Contour area:"+str('{:.3f}'.format(area_mm)) + "mm^2", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.drawContours(image=image_contour_org1, contours=contours1, contourIdx=c-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # Stop the stopwatch / counter
    t1_stop = process_time()   
    print("Elapsed time:", t1_stop, t1_start)
    print("Elapsed time during the whole program in seconds:",t1_stop-t1_start)
    image_contour_org1  = cv2.putText( image_contour_org1 , "Process Time:"+str('{0:.3f}'.format(t1_stop-t1_start))+" Sec.", (50,550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Contour detection using thre2 channels only', image_contour_org1)
    
    if (cv2.waitKey(3) == 27):
      break
        
vid.release()
cv2.destroyAllWindows()
