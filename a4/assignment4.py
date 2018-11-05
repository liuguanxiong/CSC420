import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
np.seterr(divide='ignore')

def calculate_depth():
    directory = './data/test/results/'
    files = (f for f in os.listdir(directory) if f.endswith('left_disparity.png'))
    print(files)
    for f in files:
        depth(f, directory)
def depth(file, directory):
    f = 0.721537700
    b = 0.5327119288
    img = cv2.imread(directory + file,0)
    depth = np.zeros((img.shape[0],img.shape[1]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            depth[x,y] = b*f/img[x,y]
    cv2.imwrite('./q2a_data/depth_{}'.format(file),depth*255)


if __name__ == '__main__':
    calculate_depth()


