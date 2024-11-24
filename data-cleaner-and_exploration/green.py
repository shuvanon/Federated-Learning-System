import cv2
import numpy as np
import os

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                images.append((filename, img))
    return images

def find_green_box(image):
    # Convert to HSV to better detect green color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range for green color in HSV
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming the largest green contour is the box
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, w, h)

def extract_image_inside_box(image, box):
    x, y, w, h = box
    return image[y:y+h, x:x+w]

# def resize_box(image, box, scale_factor):
#     x, y, w, h = box
#     new_w = int(w * scale_factor)
#     new_h = int(h * scale_factor)
#     center_x = x + w // 2
#     center_y = y + h // 2
#     new_x = center_x - new_w // 2
#     new_y = center_y - new_h // 2
#     return (new_x, new_y, new_w, new_h)

# def remove_green_box(image, box):
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     x, y, w, h = box
#     mask[y:y+h, x:x+w] = 255
#     inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
#     return inpainted_image

def process_and_save_images(input_directory, output_directory):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Load images
    images = load_images_from_directory(input_directory)
    
    for filename, image in images:
        # 1. Find the green box
        box = find_green_box(image)
        print(box)
        
        # 2. Extract the image inside the green box (if needed)
        extracted_image = extract_image_inside_box(image, box)
        
        # 3. Resize the box by a scale factor (e.g., 1.5) and extract the image (if needed)
        # resized_box = resize_box(image, box, 1.5)
        # resized_extracted_image = extract_image_inside_box(image, resized_box)
        
        # 4. Remove the green box
        # image_without_box = remove_green_box(image, box)
        
        # Save the processed image
        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, extracted_image)

# Directories
input_directory = "../data-cleaner-and_exploration/test_in"  
output_directory = "../data-cleaner-and_exploration/test_out"  
print("Current working directory:", os.getcwd())
# Process and save images
process_and_save_images(input_directory, output_directory)
