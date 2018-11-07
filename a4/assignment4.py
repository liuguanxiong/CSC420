import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
np.seterr(divide='ignore')

def calculate_depth():
    directory = './data/test/results/'
    files = (f for f in os.listdir(directory) if f.endswith('left_disparity.png'))
    for f in files:
        depth(f, directory)
def depth(file, directory):
    f = 721.537700
    baseline = 0.5327119288
    disparity = cv2.imread(directory + file,0)
    depth = np.divide(f*baseline,disparity,where=disparity!=0)
    cv2.imwrite('./q2a_data/depth_{}'.format(file),depth)

def convertToObject():
    files = []
    scores = []
    classes = []
    x_left = []
    y_top = []
    x_right = []
    y_bot = []
    df = pd.read_csv('./data/detections.csv')
    for index, row in df.iterrows():
        files.append(row['path'])
        scores.append(np.array([float(i) for i in row['scores'].split(',')]))
        classes.append(np.array([int(i) for i in row['classes'].split(',')]))
        x_left.append(np.array([float(i) for i in row['x_left'].split(',')]))
        y_top.append(np.array([float(i) for i in row['y_top'].split(',')]))
        x_right.append(np.array([float(i) for i in row['x_right'].split(',')]))
        y_bot.append(np.array([float(i) for i in row['y_bot'].split(',')]))
    scores = np.array(scores)
    classes = np.array(classes)
    x_left = np.array(x_left)
    y_top = np.array(y_top)
    x_right = np.array(x_right)
    y_bot = np.array(y_bot)
    return files,scores,classes,x_left,y_top,x_right,y_bot

def draw_box(threshold):
    files,scores,classes,x_left,y_top,x_right,y_bot = convertToObject()
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(files)):
        path = './data/test/left/' + files[i]
        img = cv2.imread(path)
        for b in range(len(scores[i])):
            yt = int(y_top[i][b]*img.shape[1])
            xl = int(x_left[i][b]*img.shape[0])
            yb = int(y_bot[i][b]*img.shape[1])
            xr = int(x_right[i][b]*img.shape[0])
            if classes[i][b] == 1 and scores[i][b] > threshold:
                cv2.putText(img,'person',(yt,xl),font,2,(0,165,255),1,cv2.LINE_AA)
                cv2.rectangle(img,(yt,xl),(yb,xr),(255,0,0),2)
            if classes[i][b] == 2 and scores[i][b] > threshold:
                cv2.putText(img,'bicycle',(yt,xl),font,2,(0,165,255),1,cv2.LINE_AA)
                cv2.rectangle(img,(yt,xl),(yb,xr),(0,255,0),2)
            if classes[i][b] == 3 and scores[i][b] > threshold:
                cv2.putText(img,'car',(yt,xl),font,2,(0,165,255),1,cv2.LINE_AA)
                cv2.rectangle(img,(yt,xl),(yb,xr),(0,0,255),2)
            if classes[i][b] == 10 and scores[i][b] > threshold:
                cv2.putText(img,'traffic_light',(yt,xl),font,2,(0,165,255),1,cv2.LINE_AA)
                cv2.rectangle(img,(yt,xl),(yb,xr),(255,255,0),2)
        cv2.imwrite('./q2c_data/Detect_{}'.format(files[i]),img)


def compute3D(threshold):
    f = 721.537700
    baseline = 0.5327119288
    px = 609.559300
    py = 172.854000

    files,scores,classes,x_left,y_top,x_right,y_bot = convertToObject()
    object_3D = []
    for i in range(len(files)):
        path = './data/test/left/' + files[i]
        img = cv2.imread(path)
        disparity = cv2.imread('./data/test/results/' + files[i][:-4] + '_left_disparity.png',0)
        depth = np.divide(f * baseline,disparity,where=disparity!=0)
        dic = {'person':[],'bicycle':[],'car':[],'traffic_light':[]}
        for b in range(len(scores[i])):
            yt = int(y_top[i][b]*img.shape[1])
            xl = int(x_left[i][b]*img.shape[0])
            yb = int(y_bot[i][b]*img.shape[1])
            xr = int(x_right[i][b]*img.shape[0])
            xleft = (xl+xr)//2
            yleft = (yt+yb)//2
            Z = depth[xleft,yleft]
            X = (yleft-px)*Z/f
            Y = (xleft-py)*Z/f
            if classes[i][b] == 1  and scores[i][b] > threshold:
                dic['person'].append((X,Y,Z))
            if classes[i][b] == 2 and scores[i][b] > threshold:
                dic['bicycle'].append((X,Y,Z))
            if classes[i][b] == 3 and scores[i][b] > threshold:
                dic['car'].append((X,Y,Z))
            if classes[i][b] == 10 and scores[i][b] > threshold:
                dic['traffic_light'].append((X,Y,Z))
        object_3D.append(dic)
    return object_3D

def segmentation(threshold):
    f = 721.537700
    baseline = 0.5327119288
    px = 609.559300
    py = 172.854000
    font = cv2.FONT_HERSHEY_PLAIN

    files,scores,classes,x_left,y_top,x_right,y_bot = convertToObject()
    
    for i in range(len(files)):
        path = './data/test/left/' + files[i]
        img = cv2.imread(path)
        blank_img = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
        disparity = cv2.imread('./data/test/results/' + files[i][:-4] + '_left_disparity.png',0)
        depth = np.divide(f * baseline,disparity,where=disparity!=0)
        for b in range(len(scores[i])):
            yt = int(y_top[i][b]*img.shape[1])
            xl = int(x_left[i][b]*img.shape[0])
            yb = int(y_bot[i][b]*img.shape[1])
            xr = int(x_right[i][b]*img.shape[0])
            xleft = (xl+xr)//2
            yleft = (yt+yb)//2
            Z = depth[xleft,yleft]
            X = (yleft-px)*Z/f
            Y = (xleft-py)*Z/f
            if (classes[i][b] == 1 or classes[i][b] == 2 or classes[i][b] == 3 or classes[i][b] == 10)  and scores[i][b] > threshold:
                for x in range(xl+1,xr):
                    for y in range(yt+1,yb):
                        Z_candidate = depth[x,y]
                        X_candidate = (y-px)*Z_candidate/f
                        Y_candidate = (x-py)*Z_candidate/f
                        if np.linalg.norm(np.array([X-X_candidate,Y-Y_candidate,Z-Z_candidate])) <= 3:
                            blank_img[x,y] = img[x,y]
            if classes[i][b] == 1  and scores[i][b] > threshold:
                cv2.putText(blank_img,'person',(yt,xl),font,2,(0,165,255),1,cv2.LINE_AA)
                cv2.rectangle(blank_img,(yt,xl),(yb,xr),(255,0,0),2)
            if classes[i][b] == 2 and scores[i][b] > threshold:
                cv2.putText(blank_img,'bicycle',(yt,xl),font,2,(0,165,255),1,cv2.LINE_AA)
                cv2.rectangle(blank_img,(yt,xl),(yb,xr),(0,255,0),2)
            if classes[i][b] == 3 and scores[i][b] > threshold:
                cv2.putText(blank_img,'car',(yt,xl),font,2,(0,165,255),1,cv2.LINE_AA)
                cv2.rectangle(blank_img,(yt,xl),(yb,xr),(0,0,255),2)
            if classes[i][b] == 10 and scores[i][b] > threshold:
                cv2.putText(blank_img,'traffic_light',(yt,xl),font,2,(0,165,255),1,cv2.LINE_AA)
                cv2.rectangle(blank_img,(yt,xl),(yb,xr),(255,255,0),2)
        # plt.imshow(blank_img),plt.show()
        cv2.imwrite('./q2e_data/Segmentation_{}'.format(files[i]),blank_img)
       

def text_des(object_3D):
    files,scores,classes,x_left,y_top,x_right,y_bot = convertToObject()
    for i in range(len(object_3D)):
        if files[i] == '004945.jpg' or files[i] == '004964.jpg' or files[i] == '005002.jpg':
            print('===========================Result for {}============================'.format(files[i]))
            print(' ')
            for key in object_3D[i]:
                for tup in object_3D[i][key]:
                    X, Y, Z = tup
                    dist = np.linalg.norm(np.array([X,Y,Z]))
                    if X >= 0:
                        print('There is a {} {} meters to your right'.format(key,round(np.abs(X),1)))
                    else:
                        print('There is a {} {} meters to your left'.format(key,round(np.abs(X),1)))
                    print('It is {} meters away from you'.format(round(dist,1)))
                    print(' ')


if __name__ == '__main__':
    calculate_depth()
    draw_box(0.3)
    object_3D = compute3D(0.3)
    text_des(object_3D)
    segmentation(0.3)

