# coding:utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2
import glob
import pickle

# Read image
image = mping.imread("./test_images/test1.jpg") # RGB
plt.imshow(image)

# Camera Calibration
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

# image_cal = glob.glob("./camera_cal/calibration*.jpg")

def plot2images(image1, image2, title1, title2, image1cmap=None, image2cmap='gray', save_filename=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1, cmap=image1cmap)
    ax1.set_title(title1, fontsize=30)
    ax2.imshow(image2, cmap=image2cmap)
    ax2.set_title(title2, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if save_filename:
        plt.savefig(save_filename)
    plt.show()

def rgb_select(img, r_thresh, g_thresh, b_thresh):
    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    b_channel = img[:,:,2]

    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel > g_thresh[0]) & (g_channel <= g_thresh[1])] = 1

    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel > b_thresh[0]) & (b_channel <= b_thresh[1])] = 1


luv = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2LUV)
hls = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2HLS)
hsv = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2HSV)
lab = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2LAB)

s_channel = hsv[:,:,1]
b_channel = lab[:,:,2]
l_channel = luv[:,:,0]
v_channel = hsv[:,:,2]

print s_channel.shape

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0,255)):
    # Calculate gradient magnitude

    # Convert to grayscale
    gray  = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    # Take both Sobel x and y gradients
    sobelx = cv2.sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros(gradmag)
    mag_binary[(gradmag >=mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1