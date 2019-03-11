
# coding: utf-8

# In[89]:


# import packages
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2
import glob
import pickle
get_ipython().magic('matplotlib inline')


# In[90]:


# Read in an imgae
image = mping.imread('./test_images/test1.jpg')#RGB格式
plt.imshow(image)
# image2 = cv2.imread('./test_images/test1.jpg')  #BGR格式
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(image1)
# plt.subplot(1,2,2)
# plt.imshow(image2)


# In[91]:


## camera calibration
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
image_cal = glob.glob('./camera_cal/calibration*.jpg')
# images = mping.imread('./camera_cal/calibration*.jpg')

#Step through the list and search for chessboard corners
for fname in image_cal:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        plt.imshow(img)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows() #销毁我们创建的所有窗口


# In[92]:


# Distortion correction
import pickle
get_ipython().magic('matplotlib inline')

# Test undistortion on an image
img = cv2.imread('camera_cal/calibration5.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('camera_cal/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
#以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，
#并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)


# In[ ]:





# In[93]:


image_undistorted = cv2.undistort(image, mtx, dist, None, mtx)
plt.imshow(image_undistorted)


# In[94]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(image_undistorted)
ax2.set_title('Undistorted Image', fontsize=30)


# In[95]:


# 画图
def plot2images(image1, image2, title1, title2, image1cmap=None, image2cmap='gray', save_filename=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1, cmap=image1cmap)
    ax1.set_title(title1, fontsize= 30)
    ax2.imshow(image2, cmap=image2cmap)
    ax2.set_title(title2, fontsize= 30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if save_filename:
        plt.savefig(save_filename)
    plt.show()


# In[96]:


# rgb通道
def rgb_select(img, r_thresh, g_thresh, b_thresh):
    r_channel = img[:,:,0]
    g_channel=img[:,:,1]
    b_channel = img[:,:,2]
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
    
    g_binary = np.zeros_like(g_channel)
    g_binary[(r_channel > g_thresh[0]) & (r_channel <= g_thresh[1])] = 1
    
    b_binary = np.zeros_like(b_channel)
    b_binary[(r_channel > b_thresh[0]) & (r_channel <= b_thresh[1])] = 1
    
    combined = np.zeros_like(r_channel)
    combined[((r_binary == 1) & (g_binary == 1) & (b_binary == 1))] = 1
    return combined
   


# In[97]:


plt.imshow(image_undistorted)


# In[98]:


luv= cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2LUV)
hls = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2HLS)
hsv = cv2.cvtColor(image_undistorted,cv2.COLOR_RGB2HSV)
lab=cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2LAB)
s_channel = hsv[:,:,1]
b_channel=lab[:,:,2]
l_channel = luv[:,:,0]
v_channel= hsv[:,:,2]
print(s_channel.shape)


# In[99]:



def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    #plot2images(image1=image, image2=mag_binary, title1='Original Image', title2='Thresholded Magnitude', save_filename='output_images/thresholded_magnitude.png')

    # Return the binary image
    return mag_binary


# In[100]:


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    #plot2images(image1=image, image2=grad_binary, title1='Original Image', title2='thresholded y-derivative', save_filename='output_images/thresholdedy-derivative.png')

    # Return the result
    return grad_binary


# In[101]:


def color_thresh(img, s_thresh, l_thresh, b_thresh, v_thresh):
    luv= cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    lab=cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    s_channel = hsv[:,:,1]
    b_channel=lab[:,:,2]
    l_channel = luv[:,:,0]
    v_channel= hsv[:,:,2]
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    b_binary = np.zeros_like(b_channel)
    b_binary[(s_channel > b_thresh[0]) & (s_channel <= b_thresh[1])] = 1
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(s_channel > l_thresh[0]) & (s_channel <= l_thresh[1])] = 1
    
    v_binary = np.zeros_like(v_channel)
    v_binary[(s_channel > v_thresh[0]) & (s_channel <= v_thresh[1])] = 1
    
    combined = np.zeros_like(s_channel)
    combined[((s_binary == 1) & (b_binary == 1) & (l_binary == 1) & (v_binary == 1))] = 1
    
    return combined


# In[102]:


def color_gradient_threshold(image_undistorted):
    ksize = 15
    luv= cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2LUV)
    hls = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(image_undistorted,cv2.COLOR_RGB2HSV)
    lab=cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2LAB)
#     s_channel = hsv[:,:,1]
#     b_channel=lab[:,:,2]
#     l_channel = luv[:,:,0]
#     v_channel= hsv[:,:,2]
#     mag_binary = mag_thresh(image_undistorted, sobel_kernel=ksize, mag_thresh=(150, 255))

    gradx=abs_sobel_thresh(image_undistorted,orient='x',sobel_kernel=ksize,thresh=(50,90))
    grady=abs_sobel_thresh(image_undistorted,orient='y',sobel_kernel=ksize,thresh=(30,90))
    c_binary=color_thresh(image_undistorted,s_thresh=(70,100),l_thresh=(60,255),b_thresh=(50,255),v_thresh=(150,255))
    rgb_binary=rgb_select(image_undistorted,r_thresh=(225,255),g_thresh=(225,255),b_thresh=(0,255))
    combined_binary = np.zeros_like(s_channel)

#     preprocessImage[((gradx==1) & (grady==1)) | (c_binary==1) | (rgb_binary==1)] =255
#     combined_binary[((gradx == 1) & (grady == 1) | (c_binary == 1) | (mag_binary == 1))] = 255
    combined_binary[((gradx == 1) & (grady == 1) | (c_binary == 1) | (rgb_binary==1))] = 255
    color_binary = combined_binary
    return color_binary, combined_binary
color_binary, combined_binary = color_gradient_threshold(image_undistorted )
plot2images(image1 =image , image2 = combined_binary, title1='Original Image', title2='color_gradient_threshold Image')


# In[103]:


R = image[:,:,0]
G = image[:,:,1]
B = image[:,:,2]


# In[104]:


hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]


# In[105]:


plt.imshow(S, cmap='gray')


# In[106]:


# 透视变换
def perspective_transform(image_undistorted, combined_binary):
    top_left = [560, 470]
    top_right = [730, 470]
    bottom_right = [1080, 720]
    bottom_left = [200, 720]

    top_left_dst = [200,0]
    top_right_dst = [1100,0]
    bottom_right_dst = [1100,720]
    bottom_left_dst = [200,720]
#     gray = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2GRAY)
    img_size = (image_undistorted.shape[1], image_undistorted.shape[0])
    src = np.float32([top_left,top_right, bottom_right, bottom_left] )
    dst = np.float32([top_left_dst, top_right_dst, bottom_right_dst, bottom_left_dst])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # warped =  warped = cv2.warpPerspective(image_undistorted, M, img_size)
    warped  = cv2.warpPerspective(combined_binary, M, img_size)
    return warped, Minv
warped, Minv = perspective_transform(image_undistorted, combined_binary)
plot2images(image1 = image, image2 = warped, title1='Original Image', title2='perspective_Imag')


# In[107]:


histogram = np.sum(warped[:,:], axis=0)
# 将warped中从360行开始加到720行；
#histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
plt.plot(histogram)


# In[108]:


## test
print(warped.shape[0])
#print([warped.shape[0]//2:,:])
print(np.sum(warped[warped.shape[0]//2:,:], axis=0).shape)
out_img = np.dstack((warped, warped, warped))*255
print(out_img.shape)
print(histogram.shape)
print(warped.shape)

warped2 = np.dstack((warped,warped,warped))*0.002
print(warped2.nonzero()[1])


# In[109]:


# 将warped中从360行开始加到720行；
histogram2 = np.sum(warped[warped.shape[0]//2:,:], axis=0)
out_img = np.dstack((warped, warped, warped))*255
midpoint = np.int(histogram2.shape[0]/2)

leftx_base = np.argmax(histogram2[:midpoint])
rightx_base = np.argmax(histogram2[midpoint:])+midpoint
nwindows = 1
window_height = np.int(warped.shape[0]/nwindows)
nonzero = warped.nonzero()
    
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
leftx_current = leftx_base
rightx_current = rightx_base
margin = 100
minpix = 50
left_lane_inds = []
right_lane_inds = []
    
for window in range(nwindows):
    win_y_low = warped.shape[0]-(window+1)*window_height
    win_y_high = warped.shape[0]-window*window_height
    win_xleft_low = leftx_current-margin
    win_xleft_high = leftx_current+margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
    (0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
    (0,255,0), 2) 
    
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    
    print(good_left_inds)
    
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    
    print(good_right_inds)
        
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
print(nonzero[0])
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
#     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


# In[110]:


good_left_inds.shape
#good_right_inds.shape


# In[111]:


def finding_line(warped):
    # 将warped中从360行开始加到720行；
    histogram2 = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((warped, warped, warped))*255
    midpoint = np.int(histogram2.shape[0]/2)

    leftx_base = np.argmax(histogram2[:midpoint])
    rightx_base = np.argmax(histogram2[midpoint:])+midpoint
    nwindows = 5
    window_height = np.int(warped.shape[0]/nwindows)
    nonzero = warped.nonzero()
    
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = warped.shape[0]-(window+1)*window_height
        win_y_high = warped.shape[0]-window*window_height
        win_xleft_low = leftx_current-margin
        win_xleft_high = leftx_current+margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
  
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
#     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    
        # 找出左车道线附近的像素点序号；
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
         # 找出右车道线附近的像素点序号；
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    print(left_lane_inds)
    print(right_lane_inds)

    return left_fitx, right_fitx,out_img, left_fit, right_fit,left_lane_inds,right_lane_inds
    


# In[112]:


plt.imshow(warped)


# In[113]:


left_fitx, right_fitx, out_img,left_fit, right_fit,left_lane_inds,right_lane_inds = finding_line(warped)
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )

plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
# 设置坐标轴的刻度范围
# plt.xlim(0, 1280)
# plt.ylim(720, 0)


# In[114]:


plt.imshow(warped)


# In[115]:


left_fit
right_fit


# In[116]:


binary_warped = warped


# In[117]:


nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
margin = 100
    # 找出左车道线附近的像素点序号；
left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
left_fit[1]*nonzeroy + left_fit[2] + margin))) 
     # 找出右车道线附近的像素点序号；
right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # 找到车道线像素点的位置
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
   
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # 产生一张空图
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)
    # 画图的颜色
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))


# In[118]:


def sliding_window(binary_warped,left_fit,right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    # 找出左车道线附近的像素点序号；
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
     # 找出右车道线附近的像素点序号；
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # 找到车道线像素点的位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # 重新拟合出一条二次曲线
    #left_fit = np.polyfit(lefty, leftx, 2)
    #right_fit = np.polyfit(righty, rightx, 2)
    # 产生画图的点
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # 产生一张空图
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # 画图的颜色
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
   
    return left_fitx, right_fitx, left_line_pts,right_line_pts,window_img, out_img,left_lane_inds,right_lane_inds

# Draw the lane onto the warped blank image
left_fitx, right_fitx, left_line_pts,right_line_pts,window_img, out_img, left_lane_inds, right_lane_inds =sliding_window(binary_warped,left_fit,right_fit)
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
print(left_lane_inds)
print(right_lane_inds)


# In[119]:


left_fit
right_fit


# In[120]:



def CalculateCurvature(binary_image, left_fit, right_fit, l_lane_inds, r_lane_inds):

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    img_size = (binary_image.shape[1], binary_image.shape[0])
   
    #h = binary_image.shape[0]
    ploty = np.linspace(0, img_size[1]-1, img_size[1])
    y_eval = np.max(ploty)
#     left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
#     right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48
    
    # Define conversions in x and y from pixels space to meters
    ###RESUBMIT
    # 70ft  dashed space + dashed line + dashed space
    ym_per_pix = 30/720  
#     ym_per_pix = 21.34/385 # meters per pixel in y dimension 
    # 12ft lane in 500 pixels 
    xm_per_pix = 3.7/960     # meters per pixel in y dimension 
    ### ym_per_pix = 30/720 # # meters per pixel in y dimension 
    ### xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ###RESUBMIT - END
    
    
    # 找到图像中不为零的所有像素点的像素坐标
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # 将这些不为零的像素点坐标分成x，y车道线中
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds] 
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    # 将这些像素点对应到世界坐标系中，然后拟合成二次曲线
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # 计算曲线的曲率
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # 左右车道线曲率平均
    avg_curverad = (left_curverad + right_curverad) / 2
    
## 以下计算本车在车道线中心的位置

    dist_from_center = 0.0
    # assume the camera is centered in the vehicle
    ###camera_pos = img_size[1] / 2
    if right_fit is not None:
        if left_fit is not None:
            # 摄像头位于图像中间，也是本车的中心
            camera_pos = img_size[0] / 2
            ###RESUBMIT - END
            
            # find where the right and left lanes intersect the bottom of the frame
#             left_lane_pix = np.polyval(left_fit, img_size[1])
#             right_lane_pix = np.polyval(right_fit, img_size[1])
        # 左右车道线最底端x坐标
            left_lane_pix = np.polyval(left_fit, binary_image.shape[0])
            right_lane_pix = np.polyval(right_fit, binary_image.shape[0])
        # 左右车道线中点x坐标
            center_of_lane_pix = (left_lane_pix + right_lane_pix) / 2
        # 摄像头（本车中心）与车道线中心的距离
            dist_from_center = (camera_pos - center_of_lane_pix) * 3.7/960
            #print(dist_from_center, 'm')

    return  avg_curverad, dist_from_center
# CalculateCurvature(binary_image, left_fit, right_fit, l_lane_inds, r_lane_inds)
avg_curverad, dist_from_center = CalculateCurvature(binary_warped,left_fit, right_fit, left_lane_inds, right_lane_inds)


# In[121]:


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def apply_region_of_interest_mask(image):
    x_factor = 40
    y_factor = 60
    vertices = np.array([[
        (0,image.shape[0]),
        (((image.shape[1]/2)- x_factor), (image.shape[0]/2)+ y_factor), 
         (((image.shape[1]/2) + x_factor), (image.shape[0]/2)+ y_factor), 
         (image.shape[1],image.shape[0])]], dtype=np.int32)
    #print (vertices)
    return region_of_interest(image, vertices)


# In[122]:


avg_curverad, dist_from_center


# In[123]:


def overlay_text_on_image (image, avg_curverad, dist_from_center):
    
    new_img = np.copy(image)
    #h = new_img.shape[0]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255,255,255)
    
    num_format = '{:04.2f}'
    text = 'Radius of Curvature: ' + num_format.format(avg_curverad) + 'm'
    cv2.putText(new_img, text, (40,70), font, 1.5, font_color, 2, cv2.LINE_AA)
    
    direction = 'left'
    if dist_from_center > 0:
        direction = 'right'
    abs_dist = abs(dist_from_center)
    text = 'Vehicle is ' + num_format.format(abs_dist) + ' m ' + direction + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1.5, font_color, 2, cv2.LINE_AA)
    
    return new_img
new_img = overlay_text_on_image (image, avg_curverad, dist_from_center)
plt.imshow(new_img)


# # pipline

# In[125]:


import os


# # test pipline

# In[126]:


image = mping.imread('./test_images/test2.jpg')
result = main_pipline(image)
plt.imshow(result)


# In[127]:


def read_all_images2():
    
    input = "test_images/"
    output = "output_images/"
    all_files = os.listdir(input)
    for file_name in all_files:
            #read
        images = mping.imread(input+file_name)
        result2 = main_pipline(images)
        plt.figure()
        plt.imshow(result2)
        cv2.imwrite(output + file_name, result2)
read_all_images2()
#         result2 = pipline(images)


# # Test on video
# 

# In[36]:


from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[37]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = main_pipline(image)

    return result


# In[38]:


white_output = './project_video.mp4.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')


# In[ ]:





# In[ ]:




