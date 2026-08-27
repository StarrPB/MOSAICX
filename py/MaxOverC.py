import cv2
import numpy as np
#import time
import os
#from tqdm import tqdm #not needed its a progress bar
#import matplotlib.pyplot as plt #not at all needed.

def column_horizontal_stitching( imgpath1,imgpath2,yrange,xrange):
   
    img1 = cv2.imread(imgpath1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(imgpath2, cv2.IMREAD_GRAYSCALE)
    
    images=[img1,img2]
    stitched=images[0]
    
    # Search range for matching
    
    print("Finding optimal stitching offsets...")
    for idx in range(1, len(images)):
        img2 = images[idx]
        h1, w1 = stitched.shape
        h2, w2 = img2.shape
    
        #large size canvas
        canvas = np.zeros((h1 + h2, (w1+w2+ 400)), dtype=np.uint8)
        canvas[:h1, :w1] = stitched
    
        max_overlap = -1
        best_offset = (0, 20)

        for dy in range(yrange[0], yrange[1]):
            for dx in range(xrange[0], xrange[1]):
                y1 = max(0, dy)
                x1 = max(0, dx)
                y2 = y1 + h2
                x2 = x1 + w2
    
                if y2 > canvas.shape[0] or x2 > canvas.shape[1]:
                    continue
    
                region1 = canvas[y1:y2, x1:x2]
                overlap = np.count_nonzero((region1 == 255) & (img2 == 255))
                #overlap = np.logical_and(region1, img2)
    
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_offset = (dy, dx)
    
        # Stitch using best offset
        dy, dx = best_offset
        y1 = max(0, dy)
        x1 = max(0, dx)
        y2 = y1 + h2
        x2 = x1 + w2
    
        new_canvas = np.zeros_like(canvas)
        new_canvas[:h1, :w1] = stitched
        new_canvas[y1:y2, x1:x2] = np.maximum(new_canvas[y1:y2, x1:x2], img2)
    
        # Crop black border again
        coords = cv2.findNonZero(new_canvas)
        x, y, w, h = cv2.boundingRect(coords)
        stitched = new_canvas[y:y+h, x:x+w]

        
    
    #T = np.ceil((time.time() - start_time) / 60)
    print(f"Stitched image saved")
    #print(f"Total time: {T:.2f} minutes")
    print(f"image 01 height= {h1} width= {w1}")
    print(f"image 02 height= {h2} width= {w2}" )     
    print(f"canvas height={(h1 + h2)} width={(max(w1, w2) + 400)}")
    print(f"image one placement on canvas y1={0} y2={h1} x1={0} x2={w1}")
    print(f"offsets: {(dy, dx)}")
    
    return(stitched)

###############column0 and column1###################
#start_time = time.time()

# Path to input folder

imgpath1=r'columns/column000.png' #this is the inputs for the two thingies
imgpath2=r'columns/column001.png'

#column_horizontal_stitching( imgpath1,imgpath2,y1,y2,x1,x2)
stitched=column_horizontal_stitching(imgpath1,imgpath2,[34,70],[50,150])      
cv2.imwrite("column01.png", stitched)

###############column3 and column4 ###################
#start_time = time.time()

# Path to input folder

imgpath1=r'columns/column003.png' #r'C:\Users\usd\Documents\August01\column_0_1_3.png'
imgpath2=r'columns/column004.png'


stitched=column_horizontal_stitching(imgpath1,imgpath2,[20,50],[50,150])      
cv2.imwrite("column34.png", stitched)

###############column01 and column34 ###################


def column_horizontal_stitching_No_y(imgpath1,imgpath2,x_range):
    img1 = cv2.imread(imgpath1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(imgpath2, cv2.IMREAD_GRAYSCALE)
    
    # Threshold to binary
    _, img1_bin = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    _, img2_bin = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
    
    # Match height
    h1, w1 = img1_bin.shape
    h2, w2 = img2_bin.shape
    H = max(h1, h2)
    
    if h1 < H:
        img1_bin = cv2.copyMakeBorder(img1_bin, 0, H - h1, 0, 0, cv2.BORDER_CONSTANT, value=0)
    if h2 < H:
        img2_bin = cv2.copyMakeBorder(img2_bin, 0, H - h2, 0, 0, cv2.BORDER_CONSTANT, value=0)
     
    # Create canvas >> width= image 01 and 02 width and height same as one image
    canvas_width = w1 + w2  # wide enough to avoid clipping
    canvas = np.zeros((H, canvas_width), dtype=np.uint8) 
    
    # Place image 1 at center
    img1_x = canvas_width // 2 - w1 // 2
    canvas[:, img1_x:img1_x + w1] = img1_bin
    
    # Define overlap: move image 2 so its left part overlaps image 1's right
    # Let’s start image 2 a bit before the center of image 1
    shift_x = x_range#img1_x + int(0.37 * w1)  # slight left from center
    canvas[:, shift_x:shift_x + w2] = np.maximum(canvas[:, shift_x:shift_x + w2], img2_bin)
    
    # Crop black border again
    coords = cv2.findNonZero(canvas)
    x, y, w, h = cv2.boundingRect(coords)
    stitched = canvas[y:y+h, x:x+w]
    
    print(f"image 01 height= {h1} width= {w1}")
    print(f"image 02 height= {h2} width= {w2}" )     
    print(f"canvas height={H} width={canvas_width}")
    print(f"image one placement on canvas y1={0} y2={h1} x1={img1_x} x2={img1_x + w1}")
    print(f"offsets: {(0, shift_x)}")

    return(stitched)

# Load images in grayscale
imgpath1 = r'column01.png'
imgpath2 = r'column34.png'
canvas=column_horizontal_stitching(imgpath1,imgpath2,[0,50],[200,400])
cv2.imwrite('column_0_1_3_4.png', canvas)


###############column5 and column7 ###################
#start_time = time.time()

# Path to input folder

imgpath1=r'columns/column005.png' #r'C:\Users\usd\Documents\August01\column_0_1_3.png'
imgpath2=r'columns/column007.png'


stitched=column_horizontal_stitching(imgpath1,imgpath2,[10,50],[150,300])      
cv2.imwrite("column57.png", stitched)

###############column8 and column9 ###################
#start_time = time.time()

# Path to input folder

imgpath1=r'columns/column008.png' #r'C:\Users\usd\Documents\August01\column_0_1_3.png'
imgpath2=r'columns/column009.png'

#column_horizontal_stitching( imgpath1,imgpath2,y_range,x_range)
stitched=column_horizontal_stitching(imgpath1,imgpath2,[0,50],[100,250])      
cv2.imwrite("column89.png", stitched)

###############column57 and column89 ###################
#start_time = time.time()

# Load images in grayscale
imgpath1 = r'column57.png'
imgpath2 = r'column89.png'

#column_horizontal_stitching( imgpath1,imgpath2,y_range,x_range)
stitched=column_horizontal_stitching(imgpath1,imgpath2,[0,50],[200,350])      
cv2.imwrite("column5789.png", stitched)

###############column10 and column11 ###################
#start_time = time.time()

# Path to input folder

imgpath1=r'columns/column010.png' #r'C:\Users\usd\Documents\August01\column_0_1_3.png'
imgpath2=r'columns/column011.png'

#column_horizontal_stitching( imgpath1,imgpath2,y_range,x_range)
stitched=column_horizontal_stitching(imgpath1,imgpath2,[0,50],[0,150])      
cv2.imwrite("column10_11.png", stitched)

###############column12 and column13 ###################
#start_time = time.time()

# Path to input folder

imgpath1=r'columns/column012.png' #r'C:\Users\usd\Documents\August01\column_0_1_3.png'
imgpath2=r'columns/column013.png'

#column_horizontal_stitching( imgpath1,imgpath2,y_range,x_range)
stitched=column_horizontal_stitching(imgpath1,imgpath2,[0,50],[0,150])      
cv2.imwrite("column12_13.png", stitched)

###############column10_11 and column12_13 ###################
#start_time = time.time()

# Load images in grayscale
imgpath1 = r'column10_11.png'
imgpath2 = r'column12_13.png'

#column_horizontal_stitching( imgpath1,imgpath2,y_range,x_range)
stitched=column_horizontal_stitching(imgpath1,imgpath2,[0,300],[0,300])      
cv2.imwrite("column10_11_12_13.png", stitched)

###############column0134 and column5789 ###################
#start_time = time.time()

# Load images in grayscale
imgpath1 = r'column_0_1_3_4.png'
imgpath2 = r'column5789.png'

#column_horizontal_stitching( imgpath1,imgpath2,y_range,x_range)
stitched=column_horizontal_stitching(imgpath1,imgpath2,[10,200],[500,1000])      
cv2.imwrite("column01345789.png", stitched)

###############column01345789 and column10_11_12_13 ###################
#start_time = time.time()

# Load images in grayscale
imgpath1 = r'column01345789.png'
imgpath2 = r'column10_11_12_13.png'

#column_horizontal_stitching( imgpath1,imgpath2,y_range,x_range)
stitched=column_horizontal_stitching(imgpath1,imgpath2,[470,570],[900,1100])      
cv2.imwrite("column01345789_10_11_12_13.png", stitched)

