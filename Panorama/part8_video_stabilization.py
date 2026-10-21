import time
from cv2 import cv2
import numpy as np
from pickle import dump, load


def save_homography(H, filename):
    with open(filename, "wb") as f:
        dump(H, f)


def load_homography(filename):
    with open(filename, "rb") as f:
        H = load(f)
    return H


def calculate_f(H, p_x, p_y):
    f_pow_2 = p_x * (H[0, 0]/H[2,0] + H[0, 1]/H[2, 1]) + p_y * (H[1,0]/H[2, 0] + H[1, 1]/H[2,1] - ((H[0, 0] * H[0, 1] + H[1, 0] + H[1,1]) / (H[2, 0] * H[2, 1])) - p_x ** 2 - p_y ** 2)
    return np.sqrt(f_pow_2)


def find_homography_and_Rotation(kp1, des1, kp2, des2):
    global focal_length
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])

    # H, mask = cv2.findHomography(src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, 5.0)

    X = np.transpose(src_pts)
    Y_T = dst_pts
    S = X.dot(Y_T)
    u, _, v = np.linalg.svd(S)
    det_vu = np.linalg.det(v.dot(np.transpose(u)))
    d = np.identity(2)
    d[1, 1] = det_vu
    R = np.identity(3)
    R[:2, :2] = v.dot(d).dot(np.transpose(u))
    # f = calculate_f(H, int(h / 2), int(w / 2))
    # print(f)
    # if np.isnan(f):
    #     f = focal_length
    # else:
    #     f = int(f)
    #     focal_length = f
    # k = np.asarray([[f, 0, int(h / 2)],
    #                 [0, f, int(w / 2)],
    #                 [0, 0, 1]])
    return R


def apply_homography(H, frame, pts):
    warped_pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = np.int32(warped_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(warped_pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    warped = cv2.warpPerspective(frame, Ht.dot(H), (w, h))
    return warped


def stabilize():
    frame_A = cv2.imread("Frames/image_0.jpg")
    R = np.identity(3)
    for i in range(1, 900):
        print(i)
        frame_A = apply_homography(R, frame_A, pts)
        cv2.imwrite("Stabilize/s_" + str(i - 1) + ".jpg", frame_A)
        kp_A, des_A = sift.detectAndCompute(cv2.cvtColor(frame_A, cv2.COLOR_BGR2GRAY), None)
        frame_B = cv2.imread("Frames/image_" + str(i) + ".jpg")
        kp_B, des_B = sift.detectAndCompute(cv2.cvtColor(frame_B, cv2.COLOR_BGR2GRAY), None)
        R = find_homography_and_Rotation(kp_B, des_B, kp_A, des_A)
    frame_A = apply_homography(R, frame_A, pts)
    cv2.imwrite("Stabilize/s_899.jpg", frame_A)


def create_video():
    h, w, l = frame_450.shape
    size = (w - 100, h - 100)
    out = cv2.VideoWriter("res10_video_shakeless.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30, size)
    for i in range(900):
        s_i = cv2.imread("Stabilize/s_" + str(i) + ".jpg")
        out.write(s_i[50: h - 50, 50: w - 50, :])
    out.release()


focal_length = 1000
s = time.time()
sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

frame_450 = cv2.imread("Frames/image_449.jpg")
kp_450, des_450 = sift.detectAndCompute(cv2.cvtColor(frame_450, cv2.COLOR_BGR2GRAY), None)
h, w = frame_450.shape[:2]
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

stabilize()
create_video()