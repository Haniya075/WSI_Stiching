import os
import cv2
import numpy as np
from PIL import Image

def crop_image(image, output_path):
    # Convert the PIL image to a numpy array (OpenCV image)
    image = np.array(image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to separate colorful details from the black background
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
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
    cropped_image.save(output_path)
    
    #return Image.fromarray(image), cropped_image
