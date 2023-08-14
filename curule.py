import numpy as np
import cv2 as cv
import copy
import matplotlib.pyplot as plt
import DrawCameras
import math

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    """To check if the matrix id rotation matrix
    Args:
        R (TYPE): Input matrix
    Returns:
        TYPE: Norm of image
    """
    Rt = np.transpose(R)

    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R):
    """Convert Rotation Matrix to Euler angless
    Args:
        R (TYPE): Rotation Matrix
    Returns:
        TYPE: Angles x,y,z in radians
    """
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

# compute fundamental matrix
def getFunamentalMatrix(L,R):
    
    A = np.array([[L[0][0]*R[0][0] , L[0][0]*R[0][1], L[0][0] , L[0][1]*R[0][0] , L[0][1]*R[0][1] , L[0][1] , R[0][0] , R[0][1] , 1],
                  [L[1][0]*R[1][0] , L[1][0]*R[1][1], L[1][0] , L[1][1]*R[1][0] , L[1][1]*R[1][1] , L[1][1] , R[1][0] , R[1][1] , 1],
                  [L[2][0]*R[2][0] , L[2][0]*R[2][1], L[2][0] , L[2][1]*R[2][0] , L[2][1]*R[2][1] , L[2][1] , R[2][0] , R[2][1] , 1],
                  [L[3][0]*R[3][0] , L[3][0]*R[3][1], L[3][0] , L[3][1]*R[3][0] , L[3][1]*R[3][1] , L[3][1] , R[3][0] , R[3][1] , 1],
                  [L[4][0]*R[4][0] , L[4][0]*R[4][1], L[4][0] , L[4][1]*R[4][0] , L[4][1]*R[4][1] , L[4][1] , R[4][0] , R[4][1] , 1],
                  [L[5][0]*R[5][0] , L[5][0]*R[5][1], L[5][0] , L[5][1]*R[5][0] , L[5][1]*R[5][1] , L[5][1] , R[5][0] , R[5][1] , 1],
                  [L[6][0]*R[6][0] , L[6][0]*R[6][1], L[6][0] , L[6][1]*R[6][0] , L[6][1]*R[6][1] , L[6][1] , R[6][0] , R[6][1] , 1],
                  [L[7][0]*R[7][0] , L[7][0]*R[7][1], L[7][0] , L[7][1]*R[7][0] , L[7][1]*R[7][1] , L[7][1] , R[7][0] , R[7][1] , 1],
                  ])
    
    _, _, V_t = np.linalg.svd(A)
    V = V_t.T
    F = np.reshape(V[:, -1],(3,3))
    F = np.round_(F, decimals=3)
    
    if np.linalg.matrix_rank(F) == 3: 

        U,S,V = np.linalg.svd(F)
        w = np.zeros((3,3))
        w[0,0] = S[1]
        w[1,1] = S[2]
        F = np.dot(U,np.dot(w,V))
        F = np.round_(F, decimals=3)
    return F

def bestFundamentalMatrix(list_kp1, list_kp2):
    epsilon = 0.25
    best_F = []
    highest_inlier = 0
    num_iteration = 5000
    best_kp1 = []
    best_kp2 = []
    for i in range(num_iteration):
        idx = np.random.choice(len(list_kp1) , 8 , replace=False)
        kp1 = list_kp1[idx,:]
        kp2 = list_kp2[idx,:]
        F = getFunamentalMatrix(kp1,kp2)

        inlier = 0
        for i in range(len(list_kp1)):
            x = np.array([list_kp1[i][0], list_kp1[i][1], 1])
            x_t = np.array([list_kp2[i][0], list_kp2[i][1], 1]).T
            if abs(np.dot(x_t,np.dot(F,x))) < epsilon:
                inlier += 1

        if inlier > highest_inlier:
            highest_inlier = inlier
            best_F = F
            best_kp1 = kp1
            best_kp2 = kp2      
    return best_F, best_kp1, best_kp2

def extractCameraPose(essential_matrix):
    
    R_set = []
    C_set = []
    
    U,_,V = np.linalg.svd(essential_matrix)
    
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    C1 = U[:,-1]
    R1 = np.dot(U,np.dot(W,V))
    if np.linalg.det(R1) == -1:
        R1 = -R1
        C1 = -C1
    R_set.append(R1)
    C_set.append(C1)
        
    C2 = U[:,-1]
    R2 = np.dot(U,np.dot(W.T,V))
    if np.linalg.det(R2) == -1:
        R2 = -R2
        C2 = -C2
    R_set.append(R2)
    C_set.append(C2)
        
    C3 = -U[:,-1]
    R3 = np.dot(U,np.dot(W,V))
    if np.linalg.det(R3) == -1:
        R3 = -R3
        C3 = -C3
    R_set.append(R3)
    C_set.append(C3)
        
    C4 = -U[:,-1]
    R4 = np.dot(U,np.dot(W.T,V))
    if np.linalg.det(R4) == -1:
        R4 = -R4
        C4 = -C4
    R_set.append(R4)
    C_set.append(C4)
        
    return R_set, C_set

# Visualize epilines
# def drawlines(img1src, lines, pts1src):

#     r, c,_= img1src.shape
#     np.random.seed(0)
#     for r, pt1 in zip(lines, pts1src):
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         x0, y0 = map(int, [0, -r[2]/r[1]])
#         x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
#         img1color = cv.line(img1src, (x0, y0), (x1, y1), color, 1)
#         img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
#     return img1color

def getMatchingFeature(image1, image2):
    
    des = cv.ORB_create()
    kp1, des1 = des.detectAndCompute(image1, None)
    kp2, des2 = des.detectAndCompute(image2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    lowe_ratio = 0.60

    best_matches = []
    for m,n in matches:
        if m.distance < lowe_ratio*n.distance:
            best_matches.append(m)
            
    draw_pt = []
    for m,n in matches:
        if m.distance < lowe_ratio*n.distance:
            draw_pt.append([m])
            
    list_kp1 = np.int32([kp1[mat.queryIdx].pt for mat in best_matches])
    list_kp2 = np.int32([kp2[mat.trainIdx].pt for mat in best_matches])
    final_image = cv.drawMatchesKnn(image1,kp1,image2,kp2,draw_pt,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return final_image , list_kp1, list_kp2

def calculateDisparity(image1, image2):

    disparity_map = np.zeros((image2.shape[0], image2.shape[1]), dtype=np.float64)
    
    for i in range(5,image2.shape[1]-5):
        for j in range(5, image2.shape[0]-5):
            
            E_1 = image2[j-5:j+5,i-5:i+5]
            
            SSD_max = np.inf
            corrosponding_index = 0
            for k in range(i-100, i+100, 3):
                if (k-5) < 0 or (k+5) > image1.shape[1]-1:
                    continue
                
                E_2 = image1[j-5:j+5,k-5:k+5]
                
                difference = E_1 - E_2
                difference = np.array(difference)
                element_square = np.square(difference)
                SSD = element_square.sum()
                if SSD < SSD_max:
                    SSD_max = SSD
                    corrosponding_index = k      
            disparity_map[j][i] = i - corrosponding_index
            
    disparity_map = disparity_map + np.abs(np.min(disparity_map)) + 1
    disparity_map = (disparity_map/np.max(disparity_map))*255
    disparity_map = disparity_map.astype(np.uint8)
    return disparity_map     

# def calculateDisparity(image1, image2):
    
#     disparity_map = np.zeros((image2.shape[0], image2.shape[1]), dtype=np.float64)
    
    

def computeDepthMap(image, baseline, focal_x):
    
    image = image.astype(np.float64)
    depth_map = (baseline * focal_x)/image
    depth_map[depth_map > 5000] = 5000
    depth_map = (depth_map/np.max(depth_map))*255
    depth_map = depth_map.astype(np.uint8)
    
    return depth_map 

def projection_matrix(K, R, C):
    
    A = np.concatenate((np.identity(3),-C), axis=1)
    P = np.dot(K,np.dot(R,A))
    
    return P

#Function to find 3D point given 2D pixel locations
def linear_triangulation(R_set, C_set, left_pts, right_pts, K1, K2):
    
    x3D_set = []

    for i in range(len(R_set)):
        R1, R2 = np.identity(3), R_set[i]
        C1, C2 = np.zeros((3, 1)),  C_set[i].reshape(3,1)
       
        P1 = projection_matrix(K1, R1, C1)
        P2 = projection_matrix(K2, R2, C2)
       
        p1, p2, p3 = P1
        p1_, p2_, p3_ = P2
       
        p1, p2, p3 =  p1.reshape(1,-1), p2.reshape(1,-1), p3.reshape(1,-1)
        p1_, p2_, p3_ =  p1_.reshape(1,-1), p2_.reshape(1,-1), p3_.reshape(1,-1)
       
        x3D =[]
       
        for left_pt, right_pt in zip(left_pts, right_pts):
            x, y = left_pt
            x_, y_ = right_pt
            A = np.vstack((y * p3 - p2, p1 - x * p3, y_ * p3_ - p2_, p1_-x_ * p3_ ))
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[-1]
            x3D.append(X[:3])
         
        x3D_set.append(x3D)

    return x3D_set

def disambiguateCameraPose(R_set, C_set, x3D_set):
    best_i = 0
    max_positive_depths = 0
   
    for i in range(len(R_set)):
        n_positive_depths=  0
        R, C = R_set[i],  C_set[i].reshape(-1, 1)
        r3 = R[2].reshape(1, -1)
        x3D = x3D_set[i]
       
        for X in x3D:
            X = X.reshape(-1, 1)
            if r3 @ (X - C) > 0 and X[2] > 0:
                n_positive_depths += 1
               
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths

    R, C = R_set[best_i], C_set[best_i]

    return R, C

def drawlines(img1, img2, lines, pts1, pts2):
    
    r, c,_ = img1.shape
    # print(r)
    # print(c)
      
    for r, pt1, pt2 in zip(lines, pts1, pts2):
          
        color = tuple(np.random.randint(0, 255,
                                        3).tolist())
          
        x0, y0 = map(int, [0, -r[2] / r[1] ])
        x1, y1 = map(int, 
                     [c, -(r[2] + r[0] * c) / r[1] ])
          
        img1 = cv.line(img1, 
                        (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1,
                          tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, 
                          tuple(pt2), 5, color, -1)
    return img1, img2

def main():

    curule_left = cv.imread('data/curule/im0.png',1)
    curule_left = cv.resize(curule_left, (int(curule_left.shape[1]*0.5), int(curule_left.shape[0]*0.5)))

    curule_right = cv.imread('data/curule/im1.png',1)
    curule_right = cv.resize(curule_right, (int(curule_right.shape[1]*0.5), int(curule_right.shape[0]*0.5)))

    final_image, list_kp1, list_kp2 = getMatchingFeature(curule_left, curule_right)
    # cv.imshow('Final_image', final_image)
    cv.imwrite('Feature_matching.png', final_image)
    # cv.waitKey(0)
    
    kp1 = copy.copy(list_kp1)
    kp2 = copy.copy(list_kp2)
    F, best_kp1, best_kp2 = bestFundamentalMatrix(list_kp1,list_kp2)
    # print('=======')
    # print('Fundamental Matrix')
    # print(F)
  
    K = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0,1]])
    
    # essentail matrix
    E = np.dot(np.dot(K.T,F),K)
    U,S,V = np.linalg.svd(E)
    S = np.diag(S)
    S[-1,-1] = 0
    S[1,1] = 1
    S[0,0] = 1
    E = np.dot(np.dot(U,S),V)
    # print('=======')
    # print('Essential Matrix')
    # print(E)

    R_set, C_set = extractCameraPose(E)
    
    K1 = np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0, 0, 1]])
    K2 = np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0, 0, 1]])

    x3D_set = linear_triangulation(R_set, C_set, kp1, kp2, K1, K2)
    R, C = disambiguateCameraPose(R_set, C_set, x3D_set)
    # print('=======')
    # print('camera rotation')
    # print(R)
    # print('=======')
    # print('camera translation')
    # print(C)

    # Find epipolarlines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(list_kp2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    curule_left_copy = copy.copy(curule_left)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(list_kp1.reshape(-1, 1, 2), 2, F)
    lines2 = lines2.reshape(-1, 3)
    curule_right_copy = copy.copy(curule_left)
    
    curule_left_epiline,_ = drawlines(curule_left_copy,curule_right_copy, lines1, list_kp1, list_kp2)
    curule_right_epiline,_ = drawlines(curule_right_copy, curule_left_copy, lines2, list_kp2, list_kp1)
    image = np.concatenate((curule_left_epiline,curule_right_epiline), axis =1)
    cv.imshow('Matching_Points_and_Epipolar_line.png', image)
    cv.imwrite('Matching_Points_and_Epipolar_line.png', image)
    cv.waitKey(0)

    h1, w1,_ = curule_left.shape
    h2, w2,_ = curule_right.shape

    _, H1, H2 = cv.stereoRectifyUncalibrated( np.float32(best_kp1), np.float32(best_kp2), F, imgSize=(w1, h1))
    
    curule_left_rectified_epi = cv.warpPerspective(curule_left_epiline, H1, (w1, h1))
    curule_right_rectified_epi = cv.warpPerspective(curule_right_epiline, H2, (w2, h2))
    image = np.concatenate((curule_left_rectified_epi,curule_right_rectified_epi), axis =1)
    cv.imshow('Rectified Epipolar Line', image)
    cv.imwrite('Rectified_Epipolar.png', image)
    cv.waitKey(0)
    
    curule_left_rectified = cv.warpPerspective(curule_left, H1, (w1, h1))
    curule_right_rectified = cv.warpPerspective(curule_right, H2, (w2, h2))
    
    curule_left_rect_gray = cv.cvtColor(curule_left_rectified , cv.COLOR_BGR2GRAY)
    curule_right_rect_gray = cv.cvtColor(curule_right_rectified , cv.COLOR_BGR2GRAY)
    
    disparity_map = calculateDisparity(curule_left_rect_gray, curule_right_rect_gray)
    disparity_heatmap = cv.applyColorMap(disparity_map, cv.COLORMAP_JET)
    cv.imshow('Disparity Map', disparity_map)
    cv.imwrite('Disparity_Map.png', disparity_map)
    cv.imshow('Disparity Heatmap', disparity_heatmap)
    cv.imwrite('Disparity_Heatmap.png', disparity_heatmap)
    
    baseline = 88.39
    f_x = 1758.23
    depth_map = computeDepthMap(disparity_map,baseline, f_x )
    depth_heatmap = cv.applyColorMap(depth_map, cv.COLORMAP_JET)
    cv.imshow('Depth Map', depth_map)
    cv.imwrite('Depth_Map.png', depth_map)
    cv.imshow('Depth Heatmap', depth_heatmap)
    cv.imwrite('Depth_Heatmap.png', depth_heatmap)
    
    img1 = np.hstack([disparity_map, depth_map])
    img2 = np.hstack([disparity_heatmap, depth_heatmap])
    cv.imwrite('final_map.png', img1)
    cv.imwrite('final_heatmap.png', img2)
    cv.imshow('map',img1 )
    cv.waitKey(0)
    
    
if __name__ == '__main__':
    main()