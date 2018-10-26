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

    blank = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
    h1,w1 = blank.shape[:2]
    h2,w2 = img.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, M)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    dst = cv2.warpPerspective(img, Ht.dot(M), (xmax-xmin, ymax-ymin))

    dst = dst[:-700,:-500,:]
    cv2.imwrite('./q1/transformed_door.jpg',dst)

#Question 2a
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
    result = []
    for p in percent_inlier:
        print('Affine:',np.log(1-P)/np.log(1-p**3))
        print('Homography:',np.log(1-P)/np.log(1-p**4))
        result.append((np.log(1-P)/np.log(1-p**3),np.log(1-P)/np.log(1-p**4)))
    return result

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
    cv2.imwrite("q2/ransac_affine_{}_{}".format(file1[5:-4],file2[5:]),img)
    return best_matrix

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
    cv2.imwrite("q2/ransac_homo_{}_{}".format(file1[5:-4],file2[5:]),img)
    return best_matrix

def warp_and_combine(file1,file2,M):
    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)
    img = cv2.warpPerspective(img1,M,(img2.shape[1],img2.shape[0]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (img[x,y] == [0,0,0]).all():
                img[x,y] = img2[x,y]
    cv2.imwrite("q2/using_bestM_{}".format(file2[5:]),img)

#Question 3
# I use project: https://github.com/SymenYang/Vanish-Point-Detection to help me find the vanish points,
# I can certainly extend the image and do a visual localization, but for the sake of accuracy, I decide to
# use open source algorithm to determine three vanish point. 
def find_intrinsic():
    pt1 = [-118863.0,64281.0]
    pt2 = [2151.0,1937.0]
    pt3 = [1044.0,1915.0]
    v1 = np.array(pt1+[1])
    v2 = np.array(pt2+[1])
    v3 = np.array(pt3+[1])
    A = np.array([computeRow(v1,v2),computeRow(v2,v3),computeRow(v3,v1)])
    r = np.linalg.svd(A)[2][-1]
    w = np.array([[r[0],0,r[1]],[0,r[0],r[2]],[r[1],r[2],r[3]]],dtype="float")
    K = np.linalg.inv(np.linalg.cholesky(w).T)
    K = K/K[-1:-1]
    print(K)
def computeRow(pt_a,pt_b):
    xa,ya,za = pt_a
    xb,yb,zb = pt_b
    return [xa*xb+ya*yb,xa*zb+xb*za,ya*zb+yb*za,za*zb]

#Question 4
def image_stitching(blending=True):
    img_middle = cv2.imread('./data/landscape_5.jpg')
    queue1 = []
    for i in [(1,2),(3,4),(6,7),(8,9)]:
        img1 = cv2.imread('./data/landscape_{}.jpg'.format(i[0]))
        img2 = cv2.imread('./data/landscape_{}.jpg'.format(i[1]))
        img = match_warp(img1,img2,0.5,blending=blending)
        queue1.append(img)
    queue2 = []
    for i in range(2):
        img = match_warp(queue1[2*i],queue1[2*i+1],0.5,blending=blending)
        queue2.append(img)
    img = match_warp(img_middle,queue2[0],0.5,blending=blending,img1_left=False)
    img = match_warp(img,queue2[1],0.5,blending=blending)
    if blending:
        cv2.imwrite('./q4/blending_panorama.jpg',img)
    else:
        cv2.imwrite('./q4/no_blending_panorama.jpg',img)

def match_warp(img1, img2, threshold, blending=False, img1_left=True):
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

    M, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,2.0)
  
    result = warpTwoImages(img1,img2,M,img1_left,blending)

    return result

def warpTwoImages(img1, img2, H, if_img1_left, blending):
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
    pts2_translated = cv2.perspectiveTransform(pts2,Ht.dot(H))
    img2_transformed = result.copy()
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1


    if blending:
        if if_img1_left:
            h = h1
            w = int(np.abs(t[0]+w1-pts2_[0][0][0]))+1
            left_border = int(min(t[0]+w1,pts2_[0][0][0]))
            right_border = int(max(t[0]+w1,pts2_[0][0][0]))
            alpha_mask = np.zeros((h,min(img1.shape[1],w)))
            alpha_mask[:,0:int(w/2)] = 1.0
            blur_alpha_mask = cv2.GaussianBlur(alpha_mask,(7,7),0)
            blur_alpha_mask = blur_alpha_mask[:,:,None]
            left = img1[:,-w:,:]*blur_alpha_mask
            right = img2_transformed[t[1]:t[1]+h1,left_border:right_border,:]*(1.0-blur_alpha_mask)
            result[t[1]:h1+t[1],left_border:right_border,:] = right+left
            result = result[:,:-10,:]
        else:
            h = h1
            left_border = int(t[0])
            right_border = int(pts2_translated[2][0][0])
            w = right_border - left_border
            alpha_mask = np.zeros((h,min(img1.shape[1],w)))
            alpha_mask[:,0:int(w/2)] = 1.0
            blur_alpha_mask = cv2.GaussianBlur(alpha_mask,(21,21),0)
            blur_alpha_mask = blur_alpha_mask[:,:,None]

            left = img1[:,:w,:]*(1.0-blur_alpha_mask)
            right = img2_transformed[t[1]:t[1]+h1,left_border:right_border,:]*(blur_alpha_mask)
            result[t[1]:h1+t[1],left_border:right_border,:] = right+left
            result = result[:,:-10,:]
    else:
        result = result[:,:-10,:]

    return result

if __name__ == "__main__":
    # #Question 1
    # find_h_and_w('data/door.jpeg')

    # #Question 2a  
    # image_files = ['data/im1.jpg','data/im2.jpg','data/im3.jpg']
    # for i in range(len(image_files)):
    #     matches = match('data/BookCover.jpg', image_files[i], 0.80)
    # percent_inlier = [0.5,0.5,0.5]

    #Question 2b
    # result = minimum_iterations(percent_inlier)


    #Question 2c
    # match_ransac_affine('data/BookCover.jpg','data/im1.jpg',threshold=0.80,inlier_threshold=1,k=np.ceil(result[0][0]))
    # match_ransac_affine('data/BookCover.jpg','data/im2.jpg',threshold=0.80,inlier_threshold=1,k=np.ceil(result[1][0]))
    # match_ransac_affine('data/BookCover.jpg','data/im3.jpg',threshold=0.80,inlier_threshold=1,k=np.ceil(result[2][0]))

    #Question 2d
    # M1 = match_ransac_homo('data/BookCover.jpg','data/im1.jpg',threshold=0.80,inlier_threshold=1,k=np.ceil(result[0][1]))
    # M2 = match_ransac_homo('data/BookCover.jpg','data/im2.jpg',threshold=0.80,inlier_threshold=1,k=np.ceil(result[1][1]))
    # M3 = match_ransac_homo('data/BookCover.jpg','data/im3.jpg',threshold=0.80,inlier_threshold=1,k=np.ceil(result[2][1]))
    
    #Question 2e

    # I manually set k=500 because two image has rare number of matches and I do not want to run the algorithm for too long.
    # match_ransac_homo('data/SecondBookCover.jpg','data/im1.jpg',threshold=0.80,inlier_threshold=1,k=500)
    # warp_and_combine('data/SecondBookCover.jpg','data/im1.jpg', M1)

    # match_ransac_homo('data/SecondBookCover.jpg','data/im2.jpg',threshold=0.80,inlier_threshold=1,k=500)
    # warp_and_combine('data/SecondBookCover.jpg','data/im2.jpg', M2)

    # match_ransac_homo('data/SecondBookCover.jpg','data/im3.jpg',threshold=0.80,inlier_threshold=1,k=500)
    # warp_and_combine('data/SecondBookCover.jpg','data/im3.jpg', M3)

    #Question 3
    find_intrinsic()

    #Question 4
    # image_stitching(blending=True)
    # image_stitching(blending=False)
    
    
    