# %%
import cv2
import numpy as np
import os
#from scipy.optimize import least_squares
#import matplotlib.pyplot as plt

#import itertools
#from skimage.filters import threshold_otsu, threshold_yen, threshold_triangle

# %% [markdown]
# Pre-processing image steps

# %%
def remove_border(img):
    #img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    gray = img.copy()
    
    # Create a mask where white pixels (255) are True
    white_pixels = gray == 255
    
    # Set white pixels to black (0)
    result = img.copy()
    
    result[white_pixels] = 0
    
    return result

# %%


# %%
#finding the centre of the data in a image

# %%
def positions_up(cropped_array):
    # List to store the position coordinates
    positions_up = []
    
    # Define the strip width
    strip_width = 15  # 800 / 40 = 20, so each strip is 800 rows x 20 columns
    
    # Iterate over the cropped_array in 20-column strips
    for strip in range(0, cropped_array.shape[1], strip_width):  # Iterate over columns in steps of 20
        # Extract the current strip (800 rows x 20 columns)
        strip_matrix = cropped_array[:, strip:strip+strip_width]
        
        # Flag to check if a pixel value > 200 has been found in the strip
        found = False
        
        # Scan the strip top to bottom (row-wise), starting from row 1
        for row in range(strip_matrix.shape[0]):  # Iterate over rows
            for col in range(strip_matrix.shape[1]):  # Iterate over columns within the strip
                if strip_matrix[row, col] > 200:
                    # Store the global position (row, column + strip offset)
                    positions_up.append((strip + col,row))
                    found = True
                    break  # Exit the inner loop after finding the first pixel > 200
            if found:
                break  # Exit the outer loop for this strip after finding the first pixel > 200
        
        # If no pixel > 200 is found in the strip, store a default value
        if not found:
            positions_up.append((-1, -1))  # Default value for strips with no pixel > 200
    # Step 1: Remove (-1, -1) points
    filtered_positions_up = [point for point in positions_up if point != (-1, -1)]
    x_up = np.array([p[1] for p in filtered_positions_up])
    y_up = np.array([p[0] for p in filtered_positions_up])     
    
    return(filtered_positions_up)    



def positions_down(cropped_array):
    # List to store the position coordinates
    positions_down = []
    
    # Define the strip width
    strip_width = 15  # 800 / 40 = 20, so each strip is 800 rows x 20 columns
    
    # Iterate over the cropped_array in 15-column strips
    for strip in range(0, cropped_array.shape[1], strip_width):  # Iterate over columns in steps of 15
        # Extract the current strip (800 rows x 15 columns)
        strip_matrix = cropped_array[:, strip:strip+strip_width]
        
        # Flag to check if a pixel value > 200 has been found in the strip
        found = False
        
        # Scan the strip bottom to top (row-wise), starting from the last row
        for row in reversed(range(strip_matrix.shape[0])):  # Iterate from bottom row to top
            for col in range(strip_matrix.shape[1]):  # Iterate over columns within the strip
                if strip_matrix[row, col] > 25:
                    # Store the global position (column + strip offset, row)
                    positions_down.append((strip + col, row))
                    found = True
                    break  # Exit the inner loop after finding the first pixel > 200
            if found:
                break  # Exit the outer loop for this strip after finding the first pixel > 200
        
        # If no pixel > 200 is found in the strip, store a default value
        if not found:
            positions_down.append((-1, -1))  # Default value for strips with no pixel > 200
            
    # Step 1: Remove (-1, -1) points
    filtered_positions_down = [point for point in positions_down if point != (-1, -1)]
    x_down = np.array([p[0] for p in filtered_positions_down])
    y_down = np.array([p[1] for p in filtered_positions_down])
      
    
    return(filtered_positions_down)



def find_circle_center(img):
    
    cropped_array=img
    points=positions_up(cropped_array)+positions_down(cropped_array)
    
    # Fit a circle to the points
    points_array = np.array(points)
    X = points_array[:, 0]
    Y = points_array[:, 1]
    
    # Assemble the A matrix and B vector
    A = np.c_[2*X, 2*Y, np.ones(X.shape[0])]
    B = X**2 + Y**2
    
    # Solve for circle parameters
    params = np.linalg.lstsq(A, B, rcond=None)[0]
    a, b, c = params
    center = [int(a), int(b)]
    radius = int(np.sqrt(c + a**2 + b**2))
    return center

# %%
#fit circle method
def make_dyn_mask(set_images):
    """
    Direct conversion of Processing's makeDynMask to Python/OpenCV
    set_images: List of grayscale images (800x800 numpy arrays)
    Returns: Dynamic mask (800x800 numpy array)
    """
    num_images = len(set_images)
    offsets = np.zeros((num_images, 2), dtype=int)
    shifted = [None] * num_images
    
    # Step 1: Find centers and compute offsets
    for i in range(num_images):
        center = find_circle_center(set_images[i])  # Assume returns (y, x)
        
        # Compute offset: how much to shift to move center to (400, 400)
        dy = 400 - center[1]
        dx = 400 - center[0]
        offsets[i] = [dy, dx]
    
        shifted[i] = np.zeros((800, 800), dtype=np.uint8)
    
        for y in range(800):
            for x in range(800):
                new_y = y + dy
                new_x = x + dx
                if 0 <= new_y < 800 and 0 <= new_x < 800:
                    shifted[i][new_y, new_x] = set_images[i][y, x]
    
    # Step 2: Create average mask
    map_array = np.zeros(800*800, dtype=float)
    
    for i in range(num_images):
        # Flatten image and add to map
        flat_img = shifted[i].flatten()
        map_array += flat_img.astype(float)
    
    # Compute average
    if num_images > 0:
        map_array /= num_images
    
    # Create mask image
    mask = np.zeros((800, 800), dtype=np.uint8)
    for i in range(800*800):
        mask.flat[i] = int(map_array[i])
    
    return mask



# %%
def Shifted_Img_Set(set_images):
    num_images = len(set_images)
    offsets = np.zeros((num_images, 2), dtype=int)
    shifted = [None] * num_images
    
    for i in range(num_images):
        center = find_circle_center(set_images[i])  # Assume returns (y, x)
        
        # Compute offset: how much to shift to move center to (400, 400)
        dy = 400 - center[1]
        dx = 400 - center[0]
        offsets[i] = [dy, dx]
    
        shifted[i] = np.zeros((800, 800), dtype=np.uint8)
    
        for y in range(800):
            for x in range(800):
                new_y = y + dy
                new_x = x + dx
                if 0 <= new_y < 800 and 0 <= new_x < 800:
                    shifted[i][new_y, new_x] = set_images[i][y, x]
    return shifted

# %%
def apply_mask_to_color_images(subjects, mask):
    results = []
    
    # Prepare mask
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()
    
    for subject in subjects:
        # Resize mask
        if subject.shape[:2] != mask_gray.shape:
            resized_mask = cv2.resize(mask_gray, (subject.shape[1], subject.shape[0]))
        else:
            resized_mask = mask_gray
        
        # Apply to each channel if color image
        if len(subject.shape) == 3:
            channels = cv2.split(subject)
            processed_channels = []
            for channel in channels:
                result = 255 - ((255 - channel.astype(int)) - (255 - resized_mask.astype(int)))
                result = np.clip(result, 0, 255).astype(np.uint8)
                processed_channels.append(result)
            result = cv2.merge(processed_channels)
        else:
            result = 255 - ((255 - subject.astype(int)) - (255 - resized_mask.astype(int)))
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        results.append(result)
    
    return results

# %%
def exact_circle_crop(img):
    """Exact replica of the Processing circleCrop function"""
    height, width = img.shape[:2]
    result = img.copy()
    
    outer = 250
    inner = 110
    
    for y in range(height):
        for x in range(width):
            # Convert to Processing-style coordinates
            px = x - 399
            py = 400 - y
            distance_sq = px*px + py*py
            
            if distance_sq > outer*outer or distance_sq < inner*inner:
                if len(img.shape) == 3:
                    result[y, x] = [255, 255, 255]  # Color white
                else:
                    result[y, x] = 255  # Grayscale white
    
    return result

# %%
def signal_boost(gray, threshold=254): #original was 253
    # Apply threshold: values above `threshold` -> 255, else -> 0
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return binary



# %%
# Folder containing your images


# Get list of image files (assuming common image extensions)
image_files = []

for i in (range(434)):
        # Global index (0 to 434)        
        path = f"data/_tmp{i:03d}.png"
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not load image {path}")
            continue
        image_files.append(img)
# Make sure we have exactly 434 images
if len(image_files) != 434:
    print(f"Warning: Found {len(image_files)} images in the folder, expected 434")


# Load and crop all images
cropped_images = []
for img_file in image_files[:434]:  # Process up to 434 images
    
    try:
        cropped_img = remove_border(img_file) #-------THIS IS WHERE THE STUFF IS HAPPENING
        cropped_images.append(cropped_img)
    except Exception as e:
        print(f"Error processing {img_file}: {str(e)}")

print(f"Successfully loaded and cropped {len(cropped_images)} images")


# %% [markdown]
# Shifting Images to the centre

# %%
shifted_images=Shifted_Img_Set(cropped_images)

# %% [markdown]
# Creating a dynamic mask for each batch

# %%
DynamicMaskList1=[make_dyn_mask(cropped_images[0:31]),make_dyn_mask(cropped_images[31:62]),make_dyn_mask(cropped_images[62:93]),
                 make_dyn_mask(cropped_images[93:124]),make_dyn_mask(cropped_images[124:155]),make_dyn_mask(cropped_images[155:186]),
                 make_dyn_mask(cropped_images[186:217]),make_dyn_mask(cropped_images[217:248]),make_dyn_mask(cropped_images[248:279]),
                make_dyn_mask(cropped_images[279:310]),make_dyn_mask(cropped_images[310:341]),make_dyn_mask(cropped_images[341:372]),
                              make_dyn_mask(cropped_images[372:403]),make_dyn_mask(cropped_images[403:434])]

    

# %% [markdown]
# Dynamic Mask Quantification

# %%
DynamicMaskList=[]
for i in DynamicMaskList1:
    masknew = np.clip(np.round(0.27, 2) * i, 0, 255).astype(np.uint8) # ratio= 0.27  was calculated in another notebook
    DynamicMaskList.append(masknew)

# %%
#14 masks save in a folder
# Define your target folder
output_dirMask = r"masks"

# Create the folder if it doesn't exist
os.makedirs(output_dirMask, exist_ok=True)


for i in range(len(DynamicMaskList)):
    filename = f"Mask{i:03d}.png"     # mask1.png ... mask14.png
    filepath = os.path.join(output_dirMask, filename)

    # Save image to the full path
    cv2.imwrite(filepath, DynamicMaskList[i])


    # plt.clf()  # clear previous plot
    # plt.imshow(DynamicMaskList[i], cmap='gray')
    # plt.plot(400, 400, marker='v', color="white", label="image center")
    # #plt.plot(center[0], center[1], marker='v', color="yellow", label="circle fit center")
    # #plt.grid()
    # #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.title(f"Dynamic Mask {i+1}")
    # plt.pause(1)  # pause for 1 second

# %% [markdown]
# Shifting images are divided into 14 batches

# %%
ShiftedImageList=[shifted_images[0:31],shifted_images[31:62],shifted_images[62:93],shifted_images[93:124],
                  shifted_images[124:155],shifted_images[155:186],shifted_images[186:217],shifted_images[217:248],
                  shifted_images[248:279],shifted_images[279:310],shifted_images[310:341],shifted_images[341:372],
                  shifted_images[372:403],shifted_images[403:434]]

# %%
#shifted Images save in a folder
# Define your target folder
output_dirshift = r"shiftedImages"

# Create the folder if it doesn't exist
os.makedirs(output_dirshift, exist_ok=True)


for j in range(len(shifted_images)):
    # Global index (0 to 434)
    filename = f"shifted{j:03d}.png"     # pros000.png ... pros434.png
    filepath = os.path.join(output_dirshift, filename)

    # Save image to the full path
    cv2.imwrite(filepath, shifted_images[j])

    # Optional display for debug
    # plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    # plt.title(f"Saved: {filename}")
    # plt.axis('off')
    # plt.show()


# %%
#circle cropped and signal boost Images save in a folder
# Define your target folder
output_dir = r"processedImages" #change directory according to your path
output_dir1 = r"maskedImages"
output_dir2 = r"signalBoostImages"
# Create the folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir1, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)


for j in range(14):
    results0 = apply_mask_to_color_images(ShiftedImageList[j], DynamicMaskList[j])
    for i in range(31):
        a = signal_boost(results0[i])         # Apply signal boosting
        cropped = exact_circle_crop(a)        # Apply circle crop

        index = j * 31 + i                    # Global index (0 to 434)
        
        filename = f"pros{index:03d}.png"     # pros000.png ... pros434.png
        filepath = os.path.join(output_dir, filename)
        # Save image to the full path
        cv2.imwrite(filepath, cropped)

        filename2 = f"signal_boost{index:03d}.png"     # pros000.png ... pros434.png
        filepath2 = os.path.join(output_dir2, filename2)
        # Save image to the full path
        cv2.imwrite(filepath2, a)

        filename1 = f"Masked_Shifted_Images{index:03d}.png"     # pros000.png ... pros434.png
        filepath1 = os.path.join(output_dir1, filename1)
        # Save image to the full path
        cv2.imwrite(filepath1, results0[i])

        # Optional display for debug
        # plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        # plt.title(f"Saved: {filename}")
        # plt.axis('off')
        # plt.show()


# %%
#draw graph for centre and offsets

# %%
num_images = len(cropped_images)

center1=[]
centershift1=[]
shifted_images=Shifted_Img_Set(cropped_images)
 # Step 1: Find centers and compute offsets
for i in range(num_images):
    # Find center (assuming findCenter is implemented elsewhere)
    center = find_circle_center(cropped_images[i])
    center1.append(center)

    centershift = find_circle_center(shifted_images[i])
    centershift1.append(centershift)

# %%
# Unpack x and y for both lists
x1, y1 = zip(*center1)
x2, y2 = zip(*centershift1)

# Plot both lists
#plt.figure(figsize=(8, 5))
#plt.plot(x1, y1, 'o', color='blue', label='Before')
#plt.plot(x2, y2, '*', color='red', label='After')  # 's' for square marker, dashed line
#plt.title("Data center before and after shifted")
#plt.xlabel("X")
#plt.ylabel("Y")
#plt.grid(True)
#plt.gca().invert_yaxis()  # Invert Y for image-style coordinates
#plt.legend()
#plt.tight_layout()
#plt.show()

# %%



