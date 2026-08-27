import os
import cv2
import numpy as np


def denoise_image(img1_path):
    
    img = cv2.imread(img1_path,cv2.IMREAD_GRAYSCALE)
    kernel = 255*np.ones((5,8),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #dilation = cv2.dilate(opening,kernel,iterations = 1)
    
    return opening #dilation

def process_images(images):
    max_w, max_h = 0, 0
    output_folder = "Finals"  
    os.makedirs(output_folder, exist_ok=True)
    # First pass: find max width and height
    for img in images:
        h, w = img.shape[:2]
        max_w = max(max_w, w)
        max_h = max(max_h, h)

    # Second pass: pad and save images
    n=len(images)
    for i in range(n):
        h, w = images[i].shape[:2]
        pad_top = (max_h - h) // 2
        pad_bottom = max_h - h - pad_top
        pad_left = (max_w - w) // 2
        pad_right = max_w - w - pad_left
        padded_img = cv2.copyMakeBorder(
            images[i], pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        out_path = os.path.join(output_folder, f"Final.png")
        cv2.imwrite(out_path, padded_img)

if __name__ == "__main__":
    print("Loading 14 images...")
    image_list = []
    for i in range(1):
        path = f"noisy_Final"
        #path = f'C:/Users/user/Documents/June26/Stitched_column_Img_python/f column{i:03d}.png' 
        #C:\Users\usd\Documents\June26\Stitched_column_Img_python
        img = denoise_image(path)
        if img is None:
            print(f"Warning: Could not load image {path}")
            continue
        image_list.append(img)

    if not image_list:
        print("No images loaded!")
    else:
        process_images(image_list)
        print("Images saved to output_images/")
