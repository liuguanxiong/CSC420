import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import eigh


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
    plt.imshow(img),plt.show()
    # cv2.imwrite('q2/match_threshold_{}_{}'.format(threshold,file2[5:]), img)
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

    img = cv2.warpAffine(img1,best_matrix,(img2.shape[1],img2.shape[0]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (img[x,y] == [0,0,0]).all():
                img[x,y] = img2[x,y]
    plt.imshow(img),plt.show()
    return len(good)

#Question 2d
def match_ransac_homo(file1,file2,threshold,inlier_threshold,k):
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
        maybe_inliers_idx = np.random.choice(len(src_pts),4,replace=False)
        maybe_inliers_src = src_pts[maybe_inliers_idx]
        maybe_inliers_dst = dst_pts[maybe_inliers_idx]
        matrix = cv2.getPerspectiveTransform(maybe_inliers_src,maybe_inliers_dst)
        num_of_inliers = 0
        for i in range(len(src_pts)):
            transformed = cv2.perspectiveTransform(np.array([src_pts[i]],dtype=np.float32),matrix)
            distance = np.linalg.norm(transformed-dst_pts[i][0])
            if distance < inlier_threshold:
                num_of_inliers += 1
        performance = num_of_inliers/len(src_pts)
        if performance > best_performance:
            best_performance = performance
            best_matrix = matrix
        iteration += 1
        
    img = cv2.warpPerspective(img1,best_matrix,(img2.shape[1],img2.shape[0]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (img[x,y] == [0,0,0]).all():
                img[x,y] = img2[x,y]
    plt.imshow(img),plt.show()
    return len(good)

#Question 3
def image_stitching(img1, img2, threshold):
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

    good = sorted(good, key=lambda x:x.distance)
    dst_pts= np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)

    # result = cv2.warpPerspective(img2, M, (img2.shape[1]+img1.shape[1],max(img2.shape[0],img1.shape[0])))
    # plt.imshow(result),plt.show()
    # result[0:img1.shape[0],0:img1.shape[1]] = img1
    # plt.imshow(result),plt.show()

    # result = combine_images(img1,img2,M)
    # result = mix_match(img1,result)
    result = warpTwoImages(img1,img2,M)

    return result

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))

    plt.subplot(121),plt.imshow(img1),plt.title('Input')
    plt.subplot(122),plt.imshow(result),plt.title('Output')
    plt.show()

    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

def mix_match(leftImage, warpedImage):
    i1y, i1x = leftImage.shape[:2]
    i2y, i2x = warpedImage.shape[:2]
    for i in range(0, i1x):
        for j in range(0, i1y):
            try:
                if not np.array_equal(leftImage[j,i],np.array([0,0,0])):
                    bl,gl,rl = leftImage[j,i]                               
                    warpedImage[j, i] = [bl,gl,rl]
                # if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  \
                #     np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
                #     # print "BLACK"
                #     # instead of just putting it with black, 
                #     # take average of all nearby values and avg it.
                #     warpedImage[j,i] = [0, 0, 0]
                # else:
                #     if(np.array_equal(warpedImage[j,i],[0,0,0])):
                #         # print "PIXEL"
                #         warpedImage[j,i] = leftImage[j,i]
                #     else:
                #         if not np.array_equal(leftImage[j,i], [0,0,0]):
                #             bl,gl,rl = leftImage[j,i]                               
                #             warpedImage[j, i] = [bl,gl,rl]
            except:
                pass
    # cv2.imshow("waRPED mix", warpedImage)
    # cv2.waitKey()
    return warpedImage

def combine_images(img0, img1, h_matrix):
    points0 = np.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=np.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = np.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img0.shape[0]], [img1.shape[1], 0]], dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = cv2.perspectiveTransform(points1, h_matrix)
    points = np.concatenate((points0, points2), axis=0)
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    output_img = cv2.warpPerspective(img1, H_translation.dot(h_matrix), (x_max - x_min, y_max - y_min))
    output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
    return output_img

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
    # match_ransac_affine('data/BookCover.jpg','data/im3.jpg',threshold=0.85,inlier_threshold=5,k=50)

    #Question 2d
    # match_ransac_homo('data/BookCover.jpg','data/im3.jpg',threshold=0.85,inlier_threshold=5,k=50)
    
    #Question 3
    img = cv2.imread('data/landscape_1.jpg')
    for i in range(2,7):
        print(i)
        img2 = cv2.imread('data/landscape_{}.jpg'.format(i))
        img = image_stitching(img,img2,0.3)
        cv2.imwrite('./test1.jpg',img)
    # im = cv2.imread('./test.jpg')
    # ik = cv2.imread('./data/landscape_3.jpg')
    # ij = image_stitching(im,ik,0.3)
    # cv2.imwrite('./test.jpg',ij)
    