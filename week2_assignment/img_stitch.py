# 1. Find feature points in each image
# 2. Use RANSAC to find keypoint matches
# 3. Use homography matrix to get transferring info
# 4. Merge two images

import numpy as np
import cv2

def img_stitch(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.shape[0]==3 else img1
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.shape[0]==3 else img2
    # SIFT feature points
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)  # none mask
    kp2, des2 = sift.detectAndCompute(img2, None)  # none mask
    # draw key points
    img_kp1 = cv2.drawKeypoints(img1,kp1,np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_kp2 = cv2.drawKeypoints(img2,kp2,np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #img_kps = np.concatenate((img_kp1, img_kp2), axis=1)
    #cv2.imshow('key points', img_kps)
    # find feature matches
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # draw matches
    img_match = cv2.drawMatches(img1,kp1,img2,kp2,matches[0:9],None,flags=2)
    cv2.imshow('matches', img_match)
    # get homography by RANSAC
    good_matches = matches[0:9]
    pts1 = ()
    pts2 = ()
    for m in good_matches:
        pts1 += kp1[m.queryIdx].pt
        pts2 += kp2[m.trainIdx].pt
    pts1 = np.reshape(pts1, (-1,1,2))
    pts2 = np.reshape(pts2, (-1,1,2))
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0, None, 2000, 0.8)
    print(H)
    # merge images
    dst = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    cv2.imshow("original_image_stitched", dst)
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    cv2.imshow("original_image_stitched_img2", dst)
    # image blending




if __name__=='__main__':
    img1 = cv2.imread('/Users/jinger/Downloads/Hill1.jpg')
    img2 = cv2.imread('/Users/jinger/Downloads/Hill2.jpg')
    #img = np.concatenate((img1, img2), axis=1)
    cv2.imshow('original image1', img1)
    cv2.imshow('original image2', img2)
    img_stitch(img1, img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()