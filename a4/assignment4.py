import numpy as np
import cv2

def depth(file):
    f = 0.721537700
    b = 0.5327119288
    img = cv2.imread(file)
    depth = np.zeros((img.shape[0],img.shape[1]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            depth[x,y] = b*f/img[x,y]
    return depth

