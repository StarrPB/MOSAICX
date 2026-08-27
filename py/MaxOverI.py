# %%
import cv2
import numpy as np
#from tqdm import tqdm  # for progress bars
import os
#import time
import math

# %%
def find_stitch(prior, current):
    h, w = prior.shape
    prior_mask = (prior == 0).astype(np.uint8)
    current_mask = (current == 0).astype(np.uint8)

    best_score = -1
    best_offset = [22, 55]

    # Loop through all possible offsets
    for y in range(10, 800):
        for x in range(0, 800):
            # Calculate placement range on canvas
            canvas_h = h+100
            canvas_w = w+100

            # Prior will always be placed at bottom-left of canvas
            canvas_prior = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            canvas_prior[-h:, :w] = prior_mask

            # Calculate position to place the current image
            y_start = canvas_h - h - y #900-800-y(search vertical positions) finds the height of the top left corner of the thing
            x_start = x
            y_end = y_start + h #
            x_end = x_start + w #

            # Skip if current image doesn't fit in canvas
            if y_start < 0 or x_start < 0 or y_end > canvas_h or x_end > canvas_w:
                continue

            canvas_current = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            canvas_current[y_start:y_end, x_start:x_end] = current_mask

            # Overlap where both have black pixels (1)
            overlap = np.logical_and(canvas_prior, canvas_current)
            score = np.sum(overlap)

            if score > best_score:
                best_score = score
                best_offset = [x, y]

    return best_offset



def brute_force_fit(image_list):
    """Optimized stitching function"""
    N = len(image_list)
    offsets = np.zeros((N, 2), dtype=int)
    
    print("Finding optimal stitching offsets...")
    for i in (range(1, N)):
        offsets[i] = find_stitch(image_list[i-1], image_list[i])
    
    # Calculate total canvas size needed
    total_offset = np.cumsum(offsets, axis=0)
    max_offset = np.max(total_offset, axis=0)
    min_offset = np.min(total_offset, axis=0)
    
    canvas_w = image_list[0].shape[1] + (max_offset[0] - min_offset[0]) + 10
    canvas_h = image_list[0].shape[0] + (max_offset[1] - min_offset[1]) + 10
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    print("Stitching images...")
    for i in (range(N)):
        img = image_list[i]
        x_offset, y_offset = total_offset[i]
        
        # Calculate positions
        y_pos = canvas_h - img.shape[0] - y_offset
        x_pos = x_offset
        
        # Create mask for dark pixels
        mask = img < 200
        
        # Calculate valid positions
        y_start = max(y_pos, 0)
        y_end = min(y_pos + img.shape[0], canvas_h)
        x_start = max(x_pos, 0)
        x_end = min(x_pos + img.shape[1], canvas_w)
        
        # Only proceed if there's overlap
        if y_start < y_end and x_start < x_end:
            # Calculate corresponding image region
            img_y_start = y_start - y_pos
            img_y_end = img_y_start + (y_end - y_start)
            img_x_start = x_start - x_pos
            img_x_end = img_x_start + (x_end - x_start)
            
            # Apply mask and set pixels
            canvas[y_start:y_end, x_start:x_end] = np.where(
                mask[img_y_start:img_y_end, img_x_start:img_x_end],
                255,
                canvas[y_start:y_end, x_start:x_end]
            )
    
    print("Stitching complete!")
    return canvas

# Load images efficiently
# print("Loading images...")
# image_list = []
# for i in tqdm(range(31), desc="Loading images"):
#     path = f"pros{i:03d}.png"
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"Warning: Could not load image {path}")
#         continue
#     image_list.append(img)

# if not image_list:
#     print("No images loaded!")
# else:
#     # Stitch images
#     stitched_column1 = brute_force_fit(image_list)
    
#     # Save result
#     cv2.imwrite("stitched_column_optimizedgood.png", stitched_column1)
#     print("Saved stitched_column_optimizedgoo.png")
#\Users\lillianrutowski\Downloads\PYTHON_Stitching\Masks_for_14Batches

# %%
# Define your target folder
output_dir = r"columns" #change directory according to your path

# Create the folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
# Load images efficiently
batchnum=0 #start any batch
for j in range(batchnum,14):
    image_list = []
    #start_time = time.time()
    print(f"Loading images batch number {j}")
    for i in (range(31)):
        index = j * 31 + i                    # Global index (0 to 434)        
        path = f"processedImages/pros{index:03d}.png"
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not load image {path}")
            continue
        image_list.append(img)
    
    if not image_list:
        print("No images loaded!")
    else:
        # Stitch images
        stitched_column1 = brute_force_fit(image_list)
        filename = f"column{j:03d}.png"     # column000.png ... column013.png
        filepath = os.path.join(output_dir, filename)
        
        # Save result
        cv2.imwrite(filepath, stitched_column1)
        print(f"Saved column{j}.png")

    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #minutes, seconds = divmod(elapsed_time, 60)
    #minutes = int(minutes)
    #seconds = math.ceil(seconds)

    #print(f"Execution time for column{j}.png: {minutes} minutes {seconds} seconds")

# %%


# %%



