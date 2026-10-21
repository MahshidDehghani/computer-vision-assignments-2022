import cv2
import numpy as np
from numpy import dot
from scipy import linalg


def compute_epipole(fundamental_matrix):
    U, S, V = linalg.svd(fundamental_matrix)
    e = V[-1]
    return e / e[2]


def save_match_visualization():
    res5 = np.zeros((max(h1, h2), w1 + w2, 3))
    res5[:h1, :w1] = im1
    res5[:h2, w1:] = im2
    for i in range(inlier_pts1.shape[0]):
        x1, y1 = inlier_pts1[i, :]
        cv2.circle(res5, (x1, y1), 2, (0, 255, 0), -1)
        x2, y2 = inlier_pts2[i, :]
        cv2.circle(res5, (x2 + w1, y2), 2, (0, 255, 0), -1)
        # cv2.line(res5, (x1, y1), (x2 + w1, y2), (0, 255, 0))
    for point in outlier_pts1:
        x, y = point
        cv2.circle(res5, (x, y), 2, (0, 0, 255), -1)
    for point in outlier_pts2:
        x, y = point
        cv2.circle(res5, (x + w1, y), 2, (0, 0, 255), -1)

    cv2.imwrite("res05.jpg", res5)


def save_epipole_visualizations():
    e_x, e_y = int(e1[0]), int(e1[1])
    res6 = np.zeros((max(h1, e_y) + 100, max(w1, e_x) + 100, 3))
    res6[:h1, :w1] = im1
    cv2.circle(res6, (e_x, e_y), 50, (0, 0, 255), -1)
    cv2.circle(res6, (e_x, e_y), 5, (255, 0, 0), -1)
    cv2.imwrite("res06.jpg", res6)

    e_x, e_y = int(e2[0]), int(e2[1])
    res7 = np.zeros((max(h2, e_y) + 100, max(w2, e_x) + 100, 3))
    res7[:h2, :w2] = im2
    cv2.circle(res7, (e_x, e_y), 50, (0, 0, 255), -1)
    cv2.circle(res7, (e_x, e_y), 5, (255, 0, 0), -1)
    cv2.imwrite("res07.jpg", res7)


def draw_epipolar_line(image, line):
    h, w = image.shape[:2]
    a, b, c = line
    point_list = []
    y = 0
    x = int(-(y * a + c) / b)
    if 0 <= x < h:
        point_list.append((y, x))
    y = w
    x = int(-(y * a + c) / b)
    if 0 <= x < h:
        point_list.append((y, x))
    x = 0
    y = int(-(x * b + c) / a)
    if 0 <= y < w:
        point_list.append((y, x))
    x = h
    y = int(-(x * b + c) / a)
    if 0 <= y < w:
        point_list.append((y, x))
    cv2.line(image, point_list[0], point_list[1], (0, 255, 0))


def save_epipolar_line_visualization():
    sample_indices = []
    for i in range(10):
        n = random.randint(0, inlier_pts1.shape[0])
        sample_indices.append(n)
    for i in sample_indices:
        x1, y1 = inlier_pts1[i, :]
        x2, y2 = inlier_pts2[i, :]
        line_in_image1 = dot(np.transpose(F), np.transpose([x2, y2, 1]))
        draw_epipolar_line(im1, line_in_image1)
        line_in_image2 = dot(F, np.transpose([x1, y1, 1]))
        draw_epipolar_line(im2, line_in_image2)
    res8 = np.zeros((max(h1, h2), w1 + w2, 3))
    res8[:h1, :w1] = im1
    res8[:h2, w1:] = im2
    cv2.imwrite("res08.jpg", res8)


im1 = cv2.imread("01.JPG")
h1, w1 = im1.shape[:2]
im2 = cv2.imread("02.JPG")
h2, w2 = im2.shape[:2]

gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
knn_matches = flann.knnMatch(des1, des2, k=2)

pts1 = []
pts2 = []

for i, (m, n) in enumerate(knn_matches):
    if m.distance < 0.8 * n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 20)
print("F = \n", F)

# inliers
inlier_pts1 = pts1[mask.ravel() == 1]
inlier_pts2 = pts2[mask.ravel() == 1]

# outliers
outlier_pts1 = pts1[mask.ravel() == 0]
outlier_pts2 = pts2[mask.ravel() == 0]

save_match_visualization()

e1 = compute_epipole(F)
e2 = compute_epipole(np.transpose(F))
print("e1 = ", e1)
print("e2 = ", e2)

save_epipole_visualizations()
save_epipolar_line_visualization()



