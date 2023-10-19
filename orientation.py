from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.filters import sobel
from scipy.stats import mode
import numpy as np 

def findEdges(bina_image):
  image_edges = sobel(bina_image)
  return image_edges


def findTiltAngle(image_edges):
    h, theta, d = hough_line(image_edges)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    
    # Ensure that angles is an array-like object
    angles = np.array(angles)
    
    # Calculate the mode of angles
    mode_angles = mode(angles)
    # Check if mode_angles contains a mode value
    if mode_angles.mode is not None:
        angle = np.rad2deg(mode_angles.mode)
    else:
        # Handle the case when mode_angles doesn't contain a mode
        angle = 0.0  # You can choose a default value
    
    if angle < 0:
        angle += 45
    else:
        angle -= 45
    
    return angle
  
def level(img): 
    edges = findEdges(img)
    angle = findTiltAngle(edges)

    img = rotate(img, angle)

    return img  
    