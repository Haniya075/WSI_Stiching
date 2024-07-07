import os
import cv2
import numpy as np
from PIL import Image

def crop_image(image, output_path):
    # Convert the PIL image to a numpy array (OpenCV image)
    image = np.array(image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Laplacian variance of the grayscale image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # If variance is too low, indicating a uniform region, adjust the threshold
    if variance < 50:
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    else:
        # Apply a binary threshold to get the detailed regions
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (descending) to get top 10 largest contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # Initialize an empty mask to draw the contours
    mask = np.zeros_like(gray)
    
    # Draw the top 10 contours on the mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    
    # Find bounding box coordinates of the masked area
    x, y, w, h = cv2.boundingRect(mask)
    
    # Crop the image to the bounding box
    cropped_image = image[y:y+h, x:x+w]
    
    # Convert the cropped image back to a PIL image
    cropped_image = Image.fromarray(cropped_image)
    
    # Save the cropped image
    #output_folder = r'D:\CheckOutput'
    #if not os.path.exists(output_folder):
        #os.makedirs(output_folder)
    
    #img_name = f'Output_Image_{count}.png'
    #output_path = os.path.join(output_folder, img_name)
    cropped_image.save(output_path)
    
    #return Image.fromarray(image), cropped_image
