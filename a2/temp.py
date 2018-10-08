# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('building.jpg')
def imshowBGR2RGB( im ):
  img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  plt.imshow(img)
  plt.axis('off')
  return

imshowBGR2RGB(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),7)
Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
plt.subplot(2,1,1), plt.imshow(Ix,cmap = 'gray')
plt.subplot(2,1,2), plt.imshow(Iy,cmap = 'gray')
IxIy = np.multiply(Ix, Iy)
Ix2 = np.multiply(Ix, Ix)
Iy2 = np.multiply(Iy, Iy)
Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10) 
plt.subplot(1,3,1), plt.imshow(Ix2_blur,cmap = 'gray')
plt.subplot(1,3,2), plt.imshow(Iy2_blur,cmap = 'gray')
plt.subplot(1,3,3), plt.imshow(IxIy_blur,cmap = 'gray')
det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
trace = Ix2_blur + Iy2_blur
plt.subplot(1,2,1), plt.imshow(det,cmap = 'gray')
plt.subplot(1,2,2), plt.imshow(trace,cmap = 'gray')
R = det - 0.05 * np.multiply(trace,trace)
plt.subplot(1,2,1), plt.imshow(img), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(R,cmap = 'gray'), plt.axis('off')