import time
import cv2
import numpy as np


def create_foreground_frames():
    for i in range(900):
        # print(i)
        f_i = cv2.imread("Frames/image_" + str(i) + ".jpg")
        b_i = cv2.imread("BackgroundFrames/b_" + str(i) + ".jpg")
        diff = np.abs(f_i[:, :, 0] - b_i[:, :, 0]) + np.abs(f_i[:, :, 1] - b_i[:, :, 1]) + np.abs(f_i[:, :, 2] - b_i[:, :, 2])
        thereshold = diff.max() * 0.9
        X, Y = np.where(diff > thereshold)
        for j in range(X.shape[0]):
            f_i[X[j], Y[j], 2] = min(255, f_i[X[j], Y[j], 2]  + 100)
        cv2.imwrite("foregroundFrames/f_" + str(i) + ".jpg", f_i)


def create_video():
    h, w, l = frame_450.shape
    size = (w, h)
    out = cv2.VideoWriter("res08_foreground_video.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30, size)
    for i in range(900):
        out.write(cv2.imread("foregroundFrames/f_" + str(i) + ".jpg"))
    out.release()


frame_450 = cv2.imread("Frames/image_449.jpg")
h, w = frame_450.shape[:2]
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
t_size = np.loadtxt("t_size.txt", dtype=int)
p_0 = cv2.imread("PanoromaFrames/p_0.jpg")
p_h, p_w = p_0.shape[:2]

y_450 = max(t_size[:30, 0])
x_450 = max(t_size[:30, 1])
create_foreground_frames()
create_video()





