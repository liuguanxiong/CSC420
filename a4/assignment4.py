import numpy as np
import cv2
import os
import pandas as pd
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

def draw_box():
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
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(files)):
        path = './data/test/left/' + files[i]
        img = cv2.imread(path)
        x_left *= img.shape[0]
        y_top *= img.shape[1]
        x_right *= img.shape[0]
        y_bot *= img.shape[1]
        print(path)
        for b in range(len(scores[i])):
            print('class:',classes[i][b])
            print('score:',scores[i][b])
            if classes[i][b] == 1 and scores[i][b] > 0.25:
                cv2.putText(img,'person',(int(y_top[i][b]),int(x_left[i][b])),font,1,(255,255,255),1,cv2.LINE_AA)
                cv2.rectangle(img,(int(y_top[i][b]),int(x_left[i][b])),(int(y_bot[i][b]),int(x_right[i][b])),(0,0,255),1)
            if classes[i][b] == 2 and scores[i][b] > 0.25:
                cv2.putText(img,'bicycle',(int(y_top[i][b]),int(x_left[i][b])),font,1,(255,255,255),1,cv2.LINE_AA)
                cv2.rectangle(img,(int(y_top[i][b]),int(x_left[i][b])),(int(y_bot[i][b]),int(x_right[i][b])),(0,255,0),1)
            if classes[i][b] == 3 and scores[i][b] > 0.25:
                print('hhhhhhhhh')
                cv2.putText(img,'car',(int(y_top[i][b]),int(x_left[i][b])),font,1,(255,255,255),1,cv2.LINE_AA)
                cv2.rectangle(img,(int(y_top[i][b]),int(x_left[i][b])),(int(y_bot[i][b]),int(x_right[i][b])),(255,0,0),1)
            if classes[i][b] == 10 and scores[i][b] > 0.25:
                cv2.putText(img,'traffic_light',(int(y_top[i][b]),int(x_left[i][b])),font,1,(255,255,255),1,cv2.LINE_AA)
                cv2.rectangle(img,(int(y_top[i][b]),int(x_left[i][b])),(int(y_bot[i][b]),int(x_right[i][b])),(0,255,255),1)
        plt.imshow(img),plt.show()




if __name__ == '__main__':
    # calculate_depth()
    draw_box()


