
import cv2
import numpy as np
import skimage.morphology
from scipy import ndimage
from scipy.signal import argrelextrema
from scipy.spatial.distance import cdist
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

      #Approximate DoGs with LoG
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
          cv2.circle(img,(y,x),int(np.sqrt(2*(DoGs[index]))),(255,0,0),1)
      
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
          cv2.circle(img,(y,x),int(np.sqrt(2*(DoGs[index]))),(0,255,0),1)
  cv2.imwrite('q1_data/LoG_{}'.format(file), img)

#Question 1d
def SURF(file, threshold):
  img = cv2.imread(file)
  surf = cv2.xfeatures2d.SURF_create(threshold)
  kp,des = surf.detectAndCompute(img,None)
  img2 = cv2.drawKeypoints(img,kp,None,(0,0,255),4)
  cv2.imwrite('q1_data/SURF_{}'.format(file), img2)


#Question 2a
def SIFT(file):
  img = cv2.imread(file)
  sift = cv2.xfeatures2d.SIFT_create()
  kp,des = sift.detectAndCompute(img,None)
  img2 = cv2.drawKeypoints(img,kp,None,(0,0,255),4)
  cv2.imwrite('q2_data/SIFT_{}'.format(file), img2)


#Question 2b
def match(file1, file2, threshold):
  img1 = cv2.imread(file1)
  img2 = cv2.imread(file2)
  sift = cv2.xfeatures2d.SIFT_create()
  kp1,d1 = sift.detectAndCompute(img1,None)
  kp2,d2 = sift.detectAndCompute(img2,None)

  dist = cdist(d1, d2, 'euclidean')
  dist_copy = dist.copy()
  good = []
  for i in range(len(dist_copy)):
    closest_idx = np.argmin(dist_copy[i])
    closest_dist = dist_copy[i][closest_idx]
    dist_copy[i][closest_idx] = np.inf
    second_idx = np.argmin(dist_copy[i])
    sec_dist = dist_copy[i][second_idx]
    if closest_dist/sec_dist <= threshold:
      good.append(cv2.DMatch(i,closest_idx,0,closest_dist))
  img = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)
  cv2.imwrite('q2_data/match_{}'.format(file1), img)

#Question 2c
def affine(threshold, k):
  img1 = cv2.imread('book.jpeg')
  img2 = cv2.imread('findbook.png')
  sift = cv2.xfeatures2d.SIFT_create()
  kp1,d1 = sift.detectAndCompute(img1,None)
  kp2,d2 = sift.detectAndCompute(img2,None)

  dist = cdist(d1, d2, 'euclidean')
  dist_copy = dist.copy()
  good = []
  for i in range(len(dist_copy)):
    closest_idx = np.argmin(dist_copy[i])
    closest_dist = dist_copy[i][closest_idx]
    dist_copy[i][closest_idx] = np.inf
    second_idx = np.argmin(dist_copy[i])
    sec_dist = dist_copy[i][second_idx]
    if closest_dist/sec_dist <= threshold:
      good.append(cv2.DMatch(i,closest_idx,0,closest_dist))
  
  p = []
  p_prime = []
  good = sorted(good, key=lambda x:x.distance)
  topk_index = good[:k]
  for match in topk_index:
    coor1 = kp1[match.queryIdx].pt
    coor2 = kp2[match.trainIdx].pt
    p.append([coor1[0],coor1[1],0,0,1,0])
    p.append([0,0,coor1[0],coor1[1],0,1])
    p_prime.append(coor2[0])
    p_prime.append(coor2[1])
  p = np.array(p)
  p_prime = np.array(p_prime)

  if len(topk_index) == 3:
    result = np.matmul(np.linalg.pinv(p), p_prime)
  else:
    result = np.matmul(np.matmul(np.linalg.pinv(np.matmul(p.T,p)),p.T),p_prime)
  return result
#Question 2d
def transform(A):
  img = cv2.imread('book.jpeg')
  before_corners = np.array([[0,0],[0,img.shape[1]-1],[img.shape[0]-1,0],[img.shape[0]-1,img.shape[1]-1]])
  after_corners = []
  for corner in before_corners:
    after_corners.append(tuple(np.matmul(np.array([[corner[1],corner[0],0,0,1,0],[0,0,corner[1],corner[0],0,1]]),A).astype(int)))
  find_book = cv2.imread('findBook.png')
  cv2.line(find_book,after_corners[0],after_corners[1],(255,0,0),2)
  cv2.line(find_book,after_corners[1],after_corners[3],(255,0,0),2)
  cv2.line(find_book,after_corners[2],after_corners[3],(255,0,0),2)
  cv2.line(find_book,after_corners[2],after_corners[0],(255,0,0),2)
  cv2.imwrite('q2_data/transform.jpg', find_book)

#Question 2e
def match_color(file1, file2, threshold):
  img1 = cv2.imread(file1)
  img2 = cv2.imread(file2)
  sift = cv2.xfeatures2d.SIFT_create()
  kp1,d1 = sift.detectAndCompute(img1,None)
  kp2,d2 = sift.detectAndCompute(img2,None)

  d1_color = np.zeros((d1.shape[0],d1.shape[1]+3))
  d2_color = np.zeros((d2.shape[0],d2.shape[1]+3))
  #add color descriptor
  for i in range(len(d1)):
    color = img1[int((kp1[i].pt)[0]),int((kp1[i].pt)[1])]
    d1_color[i] = np.concatenate((d1[i],color))
  for j in range(len(d2)):
    color = img2[int((kp2[j].pt)[0]),int((kp2[j].pt)[1])]
    d2_color[j] = np.concatenate((d2[j],color))

  dist = cdist(d1_color, d2_color, 'euclidean')
  dist_copy = dist.copy()
  good = []
  for i in range(len(dist_copy)):
    closest_idx = np.argmin(dist_copy[i])
    closest_dist = dist_copy[i][closest_idx]
    dist_copy[i][closest_idx] = np.inf
    second_idx = np.argmin(dist_copy[i])
    sec_dist = dist_copy[i][second_idx]
    if closest_dist/sec_dist <= threshold:
      good.append(cv2.DMatch(i,closest_idx,0,closest_dist))
  img = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)
  cv2.imwrite('q2_data/match_colour_{}'.format(file1), img)

#Question 3a
def plot_Svsk():
  k = np.arange(20)+1
  S = np.log(1-0.99)/np.log(1-0.7**k)
  plt.plot(k,S,linestyle='',marker='.')
  plt.xticks(k)
  plt.xlabel('k')
  plt.ylabel('S')
  plt.show()
#Question 3b
def plot_Svsp():
  p = (np.arange(5)+1)*0.1
  S = np.log(1-0.99)/np.log(1-p**5)
  plt.plot(p,S,linestyle='',marker='.')
  plt.xticks(p)
  plt.xlabel('p')
  plt.ylabel('S')
  plt.show()
#Question 3c



if __name__ == '__main__':
#Question 1a
  R = Harris('building.jpg')
  B = Brown('building.jpg')
#Question 1b
  for i in range(5, 21, 5):
    img = cv2.imread('building.jpg')
    result = nms(R,i,0.5*R.max())
    for x in range(result.shape[0]):
      for y in range(result.shape[1]):
        if result[x,y] == 255:
          cv2.circle(img,(y,x),2,(255,0,0),1)
    cv2.imwrite('q1_data/nms_R_{}.jpg'.format(i), img)

  for i in range(5, 21, 5):
    img = cv2.imread('building.jpg')
    result = nms(B,i,0.5*B.max())
    for x in range(result.shape[0]):
      for y in range(result.shape[1]):
        if result[x,y] == 255:
          cv2.circle(img,(y,x),2,(255,0,0),1)
    cv2.imwrite('q1_data/nms_B_{}.jpg'.format(i), img)

# Question 1c
  LoG('synthetic.png', 1.1, 1.3, 40, 20)
  LoG('building.jpg', 1.1, 1.3, 40, 3000)
# Question 1d
  SURF('building.jpg', 9000)
  SURF('synthetic.png', 5000)

# Question 2a
  SIFT('book.jpeg')
  SIFT('findBook.png')
# Question 2b
  match('book.jpeg','findBook.png',0.48)
# Question 2c
  for i in range(5,31,5):
    print(affine(0.48,i))
# Question 2d
  A = affine(0.48,20)
  transform(A)
# Question 2e
  match_color('colourTemplate.png','colourSearch.png',0.60)

# Question 3a
  plot_Svsk()
# Question 3b
  plot_Svsp()
# Question 3c
  print(np.log(1-0.99)/np.log(1-0.2**5))





