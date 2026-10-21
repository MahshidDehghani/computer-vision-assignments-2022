import cv2
import numpy as np
from pickle import dump, load


def load_homography(filename):
    with open(filename, "rb") as f:
        H = load(f)
    return H


def stitch():
    frame_270 = cv2.imread("Frames/image_269.jpg")
    frame_450 = cv2.imread("Frames/image_449.jpg")
    h, w = frame_450.shape[:2]
    H_270 = load_homography("Homography/H_269.h")
    t_size = np.loadtxt("t_size.txt", dtype=int)

    frame_450_rect = np.copy(frame_450)
    rect_pts_450 = np.float32([[w/2,h/2],[w/2,(4 * h)/5],[(2 * w)/3, (4 * h)/5],[(2 * w)/3,h/2]]).reshape(-1,1,2)
    cv2.polylines(frame_450_rect,  [rect_pts_450.astype(int)], True, 255, 5, cv2.LINE_AA)
    cv2.imwrite("res01_450_rect.jpg", frame_450_rect)

    frame_270_rect = np.copy(frame_270)
    rect_pts_270 = cv2.perspectiveTransform(rect_pts_450, np.linalg.inv(H_270))
    cv2.polylines(frame_270_rect,  [rect_pts_270.astype(int)], True, 255, 5, cv2.LINE_AA)
    cv2.imwrite("res02_270_rect.jpg", frame_270_rect)

    warped_270 = cv2.imread("WarpedFrames/w_269.jpg")
    t_270_0, t_270_1 = t_size[269, :2]
    size_270_w, size_270_h = t_size[269, 2:]
    warped_with_450 = np.zeros((size_270_h, size_270_w, 3))
    warped_with_450[:warped_270.shape[0], :warped_270.shape[1], :] = warped_270
    warped_with_450[t_270_1:h + t_270_1, t_270_0:w + t_270_0, :] = frame_450
    cv2.imwrite("res03_270_450_panorama.jpg", warped_with_450)


stitch()