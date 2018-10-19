import numpy as np
import cv2
import matplotlib.pyplot as plt

#Question 1

def find_h_and_w(file):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # The unit below is milimeter

    # I use Letter size paper
    paper_w = np.round(215.9)
    paper_h = np.round(279.4)

    paper_left_top = np.round([776.512,657.508])
    paper_right_top = np.round([979.538,675.508])
    paper_left_bot = np.round([759.527,961.488])
    paper_right_bot = np.round([958.503,967.505])
    door_left_top = np.round([[[46.1604,26.1899]]])
    door_right_top = np.round([[[1042.3,200.988]]])
    door_right_bot = np.round([[[920.002,2261.29]]])

    fro = np.array([paper_left_top,paper_right_top,paper_left_bot,paper_right_bot],np.float32)
    to = np.array([[0,0],[paper_w-1,0],[0,paper_h-1],[paper_w-1,paper_h-1]],np.float32)

    # Transform paper into its actual size scale and use the scale to determine door's dimension
    M = cv2.getPerspectiveTransform(fro, to)
    width = np.linalg.norm(cv2.perspectiveTransform(door_left_top.astype(np.float32),M)-cv2.perspectiveTransform(door_right_top.astype(np.float32),M))
    height = np.linalg.norm(cv2.perspectiveTransform(door_right_bot.astype(np.float32),M)-cv2.perspectiveTransform(door_right_top.astype(np.float32),M))
    print("width:{}mm".format(width))
    print("height:{}mm".format(height))  


    dst = cv2.warpPerspective(img,M,(1000,2200))
    print(dst)
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

if __name__ == "__main__":
    find_h_and_w('data/door.jpeg')