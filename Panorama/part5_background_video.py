import cv2
import numpy as np
from pickle import dump, load


def load_homography(filename):
    with open(filename, "rb") as f:
        H = load(f)
    return H


def get_coords(i):
    if i < 450:
        return x_450 - t_size[i, 1], y_450 - t_size[i, 0]
    elif i == 450:
        return x_450, y_450
    else:
        warped_i = cv2.imread("WarpedFrames/w_" + str(i) + ".jpg")
        return x_450 + t_size[i, 3] - warped_i.shape[0], y_450 + t_size[i, 2] - warped_i.shape[1]


def get_Ht(H):
    warped_pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = np.int32(warped_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(warped_pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    return Ht


def create_video():
    h, w, l = frame_450.shape
    size = (w, h)
    out = cv2.VideoWriter("res07_background_video.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30, size)
    for i in range(900):
        b_i = cv2.imread("BackgroundFrames/b_" + str(i) + ".jpg")
        out.write(b_i)
    out.release()



def create_wide_background_video_frames():
    background_panorama = cv2.imread("res06_background_panorama.jpg")
    for i in range(900):
        # print(i)
        warped_i = cv2.imread("WarpedFrames/w_" + str(i) + ".jpg")
        h_i, w_i = warped_i.shape[:2]
        x_i, y_i = get_coords(i)
        H_i = load_homography("Homography/H_" + str(i) + ".h")
        warped_background_i = background_panorama[x_i:x_i + h_i, y_i:y_i + w_i, :]

        background_i = cv2.warpPerspective(warped_background_i, np.linalg.inv(get_Ht(H_i).dot(H_i)), (w, h))
        cv2.imwrite("BackgroundFrames/b_" + str(i) + ".jpg", background_i)


frame_450 = cv2.imread("Frames/image_449.jpg")
h, w = frame_450.shape[:2]
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
t_size = np.loadtxt("t_size.txt", dtype=int)

y_450 = max(t_size[:30, 0])
x_450 = max(t_size[:30, 1])
create_wide_background_video_frames()
create_video()