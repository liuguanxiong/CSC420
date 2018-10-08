import cv2 as cv
import numpy as np
from scipy.ndimage import filters
from matplotlib import pyplot as plt
from scipy import signal
#Question 1
def imfilter(I, f, mode):
    img = cv.imread(I)
    if img is None:
        raise Exception("Error opening image")
    f = np.array(f)
    filter_size = f.shape
    if mode == 'valid':
        if img.shape[0]<filter_size[0] or img.shape[1]<filter_size[1]:
            raise Exception("image must be at least as large as filter in any dimension")
        if img.shape[2] == 3:
            result = np.zeros((img.shape[0]+filter_size[0]-1,img.shape[1]+filter_size[1]-1,3))
        else:
            result = np.zeros((img.shape[0]-filter_size[0]+1,img.shape[1]-filter_size[1]+1))
    elif mode == 'same':
        if img.shape[2] == 3:
            result = np.zeros((img.shape[0],img.shape[1],3))
            img = np.pad(img,((filter_size[0]//2,filter_size[0]//2),(filter_size[1]//2,filter_size[1]//2),(0,0)),'constant')
        else:
            result = np.zeros((img.shape[0],img.shape[1]))
            img = np.pad(img,((filter_size[0]//2,filter_size[0]//2),(filter_size[1]//2,filter_size[1]//2)),'constant')
    elif mode == 'full':
        if img.shape[2] == 3:
            result = np.zeros((img.shape[0]+filter_size[0]-1,img.shape[1]+filter_size[1]-1,3))
            img = np.pad(img,((filter_size[0]-1,filter_size[0]-1),(filter_size[1]-1,filter_size[1]-1),(0,0)),'constant')
        else:
            result = np.zeros((img.shape[0]+filter_size[0]-1,img.shape[1]+filter_size[1]-1))
            img = np.pad(img,((filter_size[0]-1,filter_size[0]-1),(filter_size[1]-1,filter_size[1]-1)),'constant')
    for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                result[x,y] = correlate(img, f, x+filter_size[0]//2, y+filter_size[1]//2)
    return result

def correlate(img, filter, x, y):
    filter_size = filter.shape
    flat_img = img[x-filter_size[0]//2: x+filter_size[0]//2 + 1, y-filter_size[1]//2: y+filter_size[1]//2 + 1].reshape(filter_size[0]*filter_size[1],-1)
    flat_filter = filter.flatten()
    return np.sum(flat_img * flat_filter[:,None],axis=0)

#Question 2
def gaussian_filter(img, sigmax, sigmay):
    a = np.zeros((3*sigmax,3*sigmay))
    a[3*sigmax//2,3*sigmay//2] = 1
    gaussian_f = filters.gaussian_filter(a,(sigmax,sigmay))
    return imfilter(img, gaussian_f[::-1,::-1],'same')

#Question 7
def d_gaussian_filter(file):
    img = cv.imread(file)
    if img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    new = filters.gaussian_filter(img,3,1)
    cv.imwrite('derivative_'+file,new)

def laplacian(file):
    img = cv.imread(file)
    img = cv.GaussianBlur(img,(1,1),0)
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    out = cv.Laplacian(img_grey,cv.CV_64F,3)
    cv.imwrite('lap1.jpg',out)

def laplacian_scipy(file):
    img = cv.imread(file)
    img = cv.GaussianBlur(img,(3,3),0)
    out = filters.gaussian_laplace(img,3)
    cv.imwrite('lap2.jpg',out)

#Question 8
def find_waldo(img, template):
    img_search = cv.imread(img,0)
    template = cv.imread(template,0)
    w, h = template.shape[::-1]
    method = eval('cv.TM_CCORR_NORMED')
    res = cv.matchTemplate(img_search,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    original = cv.imread(img)
    cv.rectangle(original,top_left, bottom_right, (255,0,0), 2)
    cv.imwrite('found.jpg',original)
    
#Question 11
def canny(file):
    threshold = [100,200,300,400,500]
    img = cv.imread('portrait.jpg')
    for i in range(len(threshold)):
        for j in range(i+1,len(threshold)):
            edge = cv.Canny(img,threshold[i],threshold[j])
            cv.imwrite('portrait_t1_{}_t2_{}.jpg'.format(threshold[i],threshold[j]),edge)
