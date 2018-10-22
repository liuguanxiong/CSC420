import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import os

root = os.path.dirname(os.path.abspath(__file__))
print(root)

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

    fro = np.array([paper_left_top,paper_right_top,paper_left_bot,paper_right_bot])
    to = np.array([[0,0],[paper_w-1,0],[0,paper_h-1],[paper_w-1,paper_h-1]])
    A = []
    for i in range(len(fro)):
        x = fro[i][0]
        y = fro[i][1]
        x_prime = to[i][0]
        y_prime = to[i][1]
        A.append([x,y,1,0,0,0,-x_prime*x,-x_prime*y,-x_prime])
        A.append([0,0,0,x,y,1,-y_prime*x,-y_prime*y,-y_prime])
    A = np.array(A)

    w, v = eigh(np.dot(A.T,A))
    min_idx = np.argmin(w)
    M = v[:,min_idx].reshape(3,3)
   
    # Transform paper into its actual size scale and use the scale to determine door's dimension
    # M = cv2.getPerspectiveTransform(fro, to)
    width = np.linalg.norm(cv2.perspectiveTransform(door_left_top.astype(np.float32),M)-cv2.perspectiveTransform(door_right_top.astype(np.float32),M))
    height = np.linalg.norm(cv2.perspectiveTransform(door_right_bot.astype(np.float32),M)-cv2.perspectiveTransform(door_right_top.astype(np.float32),M))
    print("width:{}mm".format(width))
    print("height:{}mm".format(height))  


    dst = cv2.warpPerspective(img,M,(1000,2200))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

#Question 2a
# def match(file1, file2):
#     img1 = cv2.imread(file1)
#     img2 = cv2.imread(file2)
#     img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     sift = cv2.xfeatures2d.SIFT_create()
#     kp1,d1 = sift.detectAndCompute(img1_gray,None)
#     kp2,d2 = sift.detectAndCompute(img2_gray,None)

#     # BFMatcher with default params
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(d1,d2,k=2)
#     # Apply ratio test
#     good = []
#     for m,n in matches:
#         if m.distance < 0.75*n.distance:
#             good.append([m])
#     # cv.drawMatchesKnn expects list of lists as matches.
#     img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:10],None,flags=2)
#     plt.imshow(img3),plt.show()
def match(file1, file2, threshold):
    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1,d1 = sift.detectAndCompute(gray1,None)
    kp2,d2 = sift.detectAndCompute(gray2,None)

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
    cv2.imwrite('q2/match_threshold_{}_{}'.format(threshold,file2[5:]), img)
    return len(good)


#Question 2b
def minimum_iterations(percent_inlier):
    P = 0.99
    for p in percent_inlier:
        print('Affine:',np.log(1-P)/np.log(1-p**3))
        print('Homography:',np.log(1-P)/np.log(1-p**3))


#Question 2c
def match_ransac_affine(file1,file2,threshold,inlier_threshold,k):
    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1,d1 = sift.detectAndCompute(gray1,None)
    kp2,d2 = sift.detectAndCompute(gray2,None)

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

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    iteration = 0
    best_performance = 0
    best_matrix = None
    while iteration < k:
        maybe_inliers_idx = np.random.choice(len(src_pts),3,replace=False)
        maybe_inliers_src = src_pts[maybe_inliers_idx]
        maybe_inliers_dst = dst_pts[maybe_inliers_idx]
        matrix = cv2.getAffineTransform(maybe_inliers_src,maybe_inliers_dst)
        num_of_inliers = 0
        for i in range(len(src_pts)):
            transformed = np.matmul(matrix,np.concatenate((src_pts[i][0],[1])))
            distance = np.linalg.norm(transformed-dst_pts[i][0])
            if distance < inlier_threshold:
                num_of_inliers += 1
        performance = num_of_inliers/len(src_pts)
        if performance > best_performance:
            best_performance = performance
            best_matrix = matrix
        iteration += 1
    print(best_matrix)

    img = cv2.warpAffine(img1,best_matrix,(400,400))
    plt.imshow(img),plt.show()
    return len(good)

#Question 3

if __name__ == "__main__":
    #Question 1
    # find_h_and_w('data/door.jpeg')

    #Question 2a
    # outlier = [8,11,2]
    # image_files = ['data/im1.jpg','data/im2.jpg','data/im3.jpg']
    # percent_inlier = []
    # for i in range(len(image_files)):
    #     matches = match('data/BookCover.jpg', image_files[i], 0.85)
    #     percent_outlier = outlier[i]/matches
    #     percent_inlier.append(1-percent_outlier)

    #Question 2b
    # minimum_iterations(percent_inlier)

    #Question 2c
    match_ransac_affine('data/BookCover.jpg','data/im3.jpg',threshold=0.85,inlier_threshold=5,k=50)