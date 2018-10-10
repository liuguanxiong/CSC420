
import cv2
import numpy as np
import skimage.morphology
from scipy import ndimage
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

#Question 1a
def Harris(file):
  img = cv2.imread(file)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),7)
  Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
  Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

  IxIy = np.multiply(Ix, Iy)
  Ix2 = np.multiply(Ix, Ix)
  Iy2 = np.multiply(Iy, Iy)
  Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
  Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
  IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10) 
  det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
  trace = Ix2_blur + Iy2_blur
  R = det - 0.05 * np.multiply(trace,trace)
  norm_R = cv2.normalize(R, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
  cv2.imwrite('q1_data/R.jpg',norm_R)
  return R

def Brown(file):
  img = cv2.imread(file)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),7)
  Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
  Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

  IxIy = np.multiply(Ix, Iy)
  Ix2 = np.multiply(Ix, Ix)
  Iy2 = np.multiply(Iy, Iy)
  Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
  Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
  IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10) 
  det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
  trace = Ix2_blur + Iy2_blur

  B = det/trace
  norm_B = cv2.normalize(B, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
  cv2.imwrite('q1_data/B.jpg',norm_B)
  return B


#Question 1b
def nms(img, radius, threshold):
  circle = skimage.morphology.disk(radius)
  filtered = ndimage.rank_filter(img, rank=-1, footprint=circle)
  mask = img == filtered
  result = np.where(filtered > threshold, filtered, 0)
  result = np.where(mask,255,0)
  return result

#Question 1c
def LoG(file, k, sigma, layers, threshold):
  img = cv2.imread(file)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_smooth = gray
  sigmas = (np.ones(layers) * k) ** (np.arange(layers))
  LoG = np.zeros((img.shape[0], img.shape[1], len(sigmas)))

  #Convolve the image with different laplacian of gaussian
  for i in range(len(sigmas)):
    result = ndimage.gaussian_laplace(img_smooth, sigma=sigmas[i])
    LoG[:,:,i] = sigmas[i]**2 * result
  

  #Find local extreme in both scales and space
  for x in range(1,img.shape[0]-1):
    for y in range(1,img.shape[1]-1):

      DoGs = LoG[x,y,:]
      maximas = argrelextrema(DoGs, np.greater)[0]
      minimas = argrelextrema(DoGs, np.less)[0]

      for index in maximas:
        if np.abs(DoGs[index]) < threshold:
          continue
        top_layer = LoG[x-1:x+2,y-1:y+2,index+1].flatten()
        current_layer = LoG[x-1:x+2,y-1:y+2,index].flatten()
        bottom_layer = LoG[x-1:x+2,y-1:y+2,index-1].flatten()
        cube = np.concatenate((top_layer,current_layer,bottom_layer))
        new_cube = np.delete(cube,[13])
        difference = np.ones(len(new_cube)) * DoGs[index] - new_cube
        if min(difference)>0:
          cv2.circle(img,(y,x),int(np.sqrt(2)*(index*2)),(255,0,0),1)
      
      for index in minimas:
        if np.abs(DoGs[index]) < threshold:
          continue
        top_layer = LoG[x-1:x+2,y-1:y+2,index+1].flatten()
        current_layer = LoG[x-1:x+2,y-1:y+2,index].flatten()
        bottom_layer = LoG[x-1:x+2,y-1:y+2,index-1].flatten()
        cube = np.concatenate((top_layer,current_layer,bottom_layer))
        new_cube = np.delete(cube,[13])
        difference = np.ones(len(new_cube)) * DoGs[index] - new_cube
        if max(difference)<0:
          cv2.circle(img,(y,x),int(np.sqrt(2)*(index*2)),(0,255,0),1)
  cv2.imwrite('q1_data/LoG_{}'.format(file), img)

def SURF(file, threshold):
  img = cv2.imread(file)
  surf = cv2.xfeatures2d.SURF_create(threshold)
  kp,des = surf.detectAndCompute(img,None)
  img2 = cv2.drawKeypoints(img,kp,None,(0,0,255),4)
  cv2.imwrite('q1_data/SURF_{}'.format(file), img2)

def SIFT(file, threshold):
  img = cv2.imread(file)
  sift = cv2.xfeatures2d.SIFT_create(threshold)
  kp,des = sift.detectAndCompute(img,None)
  img2 = cv2.drawKeypoints(img,kp,None,(0,0,255),4)
  cv2.imwrite('q1_data/SIFT_{}'.format(file), img2)


if __name__ == '__main__':
# #Question 1a
#   R = Harris('building.jpg')
#   B = Brown('building.jpg')
# #Question 1b
#   for i in range(5, 21, 5):
#     img = cv2.imread('building.jpg')
#     result = nms(R,i,0.5*R.max())
#     for x in range(result.shape[0]):
#       for y in range(result.shape[1]):
#         if result[x,y] == 255:
#           cv2.circle(img,(y,x),2,(255,0,0),1)
#     cv2.imwrite('q1_data/nms_R_{}.jpg'.format(i), img)

#   for i in range(5, 21, 5):
#     img = cv2.imread('building.jpg')
#     result = nms(B,i,0.5*B.max())
#     for x in range(result.shape[0]):
#       for y in range(result.shape[1]):
#         if result[x,y] == 255:
#           cv2.circle(img,(y,x),2,(255,0,0),1)
#     cv2.imwrite('q1_data/nms_B_{}.jpg'.format(i), img)
#Question 1c
  # LoG('synthetic.png', 1.1, 1.3, 40, 20)
  # LoG('building.jpg', 1.1, 1.3, 40, 2500)
  SURF('building.jpg', 9000)
  SURF('synthetic.png', 5000)
  SIFT('building.jpg', 50000)
  SIFT('synthetic.png', 200)


