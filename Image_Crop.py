import os
import cv2
import numpy as np
from PIL import Image

def crop_image(image, output_path):
    #image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(mask)
    cropped_image = image[y:y+h, x:x+w]
    cropped_image = Image.fromarray(cropped_image)
    cropped_image.save(output_path)
    print(f'Cropped image dimensions: {cropped_image.size}')

return cropped_image
    

