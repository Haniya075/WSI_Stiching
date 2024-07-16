import os
import cv2
import numpy as np
from PIL import Image

def crop_image(image, output_path):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the Laplacian variance of the grayscale image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 50:
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    x, y, w, h = cv2.boundingRect(mask)
    cropped_image = image[y:y+h, x:x+w]
    cropped_img.save(output_path)
    print(f'Cropped image dimensions: {cropped_image.size}')

    return cropped_image
    

