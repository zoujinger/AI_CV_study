import numpy as np
import cv2

def ransacMatching(A, B):
#   A & B: List of List

Algorithm for this procedure can be described like this:
#        1. Choose 4 pair of points randomly in our matching points. Those four called "inlier" (中文： 内点) while
#          others "outlier" (中文： 外点)
#       2. Get the homography of the inliers
#       3. Use this computed homography to test all the other outliers. And separated them by using a threshold
#          into two parts:
#          a. new inliers which is satisfied our computed homography
#          b. new outliers which is not satisfied by our computed homography.
#       4. Get our all inliers (new inliers + old inliers) and goto step 2
#       5. As long as there's no changes or we have already repeated step 2-4 k, a number actually can be computed,
#          times, we jump out of the recursion. The final homography matrix will be the one that we want.


## pseudo code
A = points1, B = points2, H = homography
k = iteration times
th = threshold for inlier and outlier
error = SSD(A_in, H*B_in)
total_error
best_error = minimun error
d = number of points need to be matched


for i in range (k):
    A_in, B_in = 4 pairs of points randomly
    A_out, B_out = outliers
    num_inlier = 4
    H = homography(A_in, B_in)
    total_error = 0

    for p in A_out:
        error = SSD(point_a, H*point_b)
        if error < th:
            num_inlier += 1
            A_in.append(point_a)
            B_in.append(point_b)
            A_out.delete(point_a)
            B_out.delete(point_b)
            total_error += error
    H = homography(A_in, B_in)
    if num_inlier > d and total_error < best_error:
        return H

