
import cv2
import numpy as np
import skimage.morphology
from scipy import ndimage
from scipy.signal import argrelextrema
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
  response = np.zeros((img.shape[0], img.shape[1], len(sigmas)))

  for i in range(len(sigmas)):
    result = ndimage.gaussian_laplace(img_smooth, sigma=sigmas[i])
    response[:,:,i] = sigmas[i]**2 * result
  
  for x in range(2,img.shape[0]-1):
    for y in range(2,img.shape[1]-1):
      DoGs = response[x,y,:]
      maximas = argrelextrema(DoGs, np.greater)[0]
      minimas = argrelextrema(DoGs, np.less)[0]

      for index in maximas:
        if np.abs(DoGs[index]) < threshold:
          continue
        top_layer = response[x-1:x+2,y-1:y+2,index+1].flatten()
        current_layer = response[x-1:x+2,y-1:y+2,index].flatten()
        bottom_layer = response[x-1:x+2,y-1:y+2,index-1].flatten()
        cube = np.concatenate((top_layer,current_layer,bottom_layer))
        new_cube = np.delete(cube,[13])
        difference = np.ones(len(new_cube)) * DoGs[index] - new_cube
        if min(difference)>0:
          cv2.circle(img,(y,x),int(1*DoGs[index]),(255,0,0),1)
      
      for index in minimas:
        if np.abs(DoGs[index]) < threshold:
          continue
        top_layer = response[x-1:x+2,y-1:y+2,index+1].flatten()
        current_layer = response[x-1:x+2,y-1:y+2,index].flatten()
        bottom_layer = response[x-1:x+2,y-1:y+2,index-1].flatten()
        cube = np.concatenate((top_layer,current_layer,bottom_layer))
        new_cube = np.delete(cube,[13])
        difference = np.ones(len(new_cube)) * DoGs[index] - new_cube
        if max(difference)<0:
          cv2.circle(img,(y,x),int(1*DoGs[index]),(0,255,0),1)
  cv2.imwrite('q1_data/LoG.jpg', img)

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
  # LoG('synthetic.png', 1.1, 2.0, 30, 20)

  # Read image
  im = cv2.imread("synthetic.png")

  # Setup SimpleBlobDetector parameters.
  params = cv2.SimpleBlobDetector_Params()

  # Change thresholds
  params.minThreshold = 10
  params.maxThreshold = 200


  # Filter by Area.
  params.filterByArea = True
  params.minArea = 1500

  # Filter by Circularity
  params.filterByCircularity = True
  params.minCircularity = 0.1

  # Filter by Convexity
  params.filterByConvexity = True
  params.minConvexity = 0.87

  # Filter by Inertia
  params.filterByInertia = True
  params.minInertiaRatio = 0.01

  # Create a detector with the parameters
  detector = cv2.SimpleBlobDetector(params)


  # Detect blobs.
  keypoints = detector.detect(im)

  # Draw detected blobs as red circles.
  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
  # the size of the circle corresponds to the size of blob

  im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

  # Show blobs
  cv2.imshow("Keypoints", im_with_keypoints)
  cv2.waitKey(0)
