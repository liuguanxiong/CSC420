import matplotlib.pyplot as plt
import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def Harris(file):
  img = cv2.imread(file)
  cv2.imwrite('q1_data/original.jpg',img)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),7)
  Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
  Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
  cv2.imwrite('q1_data/Ix.jpg', Ix)
  cv2.imwrite('q1_data/Iy.jpg', Iy)

  IxIy = np.multiply(Ix, Iy)
  Ix2 = np.multiply(Ix, Ix)
  Iy2 = np.multiply(Iy, Iy)
  Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
  Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
  IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10) 
  cv2.imwrite('q1_data/Ix2_blur.jpg', Ix2_blur)
  cv2.imwrite('q1_data/Iy2_blur.jpg', Iy2_blur)
  cv2.imwrite('q1_data/IxIy_blur.jpg', IxIy_blur)
  det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
  trace = Ix2_blur + Iy2_blur
  cv2.imwrite('q1_data/det.jpg',det)
  cv2.imwrite('q1_data/trace.jpg', trace)
  R = det - 0.05 * np.multiply(trace,trace)
  cv2.imwrite('q1_data/R.jpg',R)

def Brown(file):
  img = cv2.imread(file)
  cv2.imwrite('q1_data/original.jpg',img)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),7)
  Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
  Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
  cv2.imwrite('q1_data/Ix.jpg', Ix)
  cv2.imwrite('q1_data/Iy.jpg', Iy)

  IxIy = np.multiply(Ix, Iy)
  Ix2 = np.multiply(Ix, Ix)
  Iy2 = np.multiply(Iy, Iy)
  Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
  Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
  IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10) 
  cv2.imwrite('q1_data/Ix2_blur.jpg', Ix2_blur)
  cv2.imwrite('q1_data/Iy2_blur.jpg', Iy2_blur)
  cv2.imwrite('q1_data/IxIy_blur.jpg', IxIy_blur)
  det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
  trace = Ix2_blur + Iy2_blur
  cv2.imwrite('q1_data/det.jpg',det)
  cv2.imwrite('q1_data/trace.jpg', trace)

  B = det/trace
  cv2.imwrite('q1_data/B.jpg',B)

if __name__ == '__main__':
  Harris('building.jpg')
  Brown('building.jpg')