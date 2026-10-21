import time
import cv2
import numpy as np
from pickle import dump, load
import multiprocessing
from joblib import Parallel, delayed


def load_homography(filename):
    with open(filename, "rb") as f:
        H = load(f)
    return H


def save_panorama_frame_less_than_450(i):
    # print(i)
    empty_image = np.zeros((total_h, total_w, 3))
    x_i, y_i = x_450 - t_size[i, 1], y_450 - t_size[i, 0]
    warped_i = cv2.imread("WarpedFrames/w_" + str(i) + ".jpg")
    empty_image[x_i:warped_i.shape[0] + x_i, y_i:warped_i.shape[1] + y_i, :] = warped_i
    cv2.imwrite("PanoromaFrames/p_" + str(i) + ".jpg", empty_image)


def save_panorama_frame_more_than_450(i):
    # print(i)
    empty_image = np.zeros((total_h, total_w, 3))
    warped_i = cv2.imread("WarpedFrames/w_" + str(i) + ".jpg")
    x_i, y_i = x_450 + t_size[i, 3] - warped_i.shape[0], y_450 + t_size[i, 2] - warped_i.shape[1]
    empty_image[x_i:warped_i.shape[0] + x_i, y_i:warped_i.shape[1] + y_i, :] = warped_i
    cv2.imwrite("PanoromaFrames/p_" + str(i) + ".jpg", empty_image)


def create_video(video_name):
    first_image = cv2.imread("PanoromaFrames/p_0.jpg")
    h, w, l = first_image.shape
    size = (w, h)
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
    for i in range(900):
        # print(i)
        image_i = cv2.imread("PanoromaFrames/p_" + str(i) + ".jpg")
        out.write(image_i)
    out.release()


if __name__ == "__main__":
    s = time.time()
    num_cores = multiprocessing.cpu_count()
    frame_450 = cv2.imread("Frames/image_449.jpg")

    h, w = frame_450.shape[:2]
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    t_size = np.loadtxt("t_size.txt", dtype=int)

    y_450 = max(t_size[:30, 0])
    x_450 = max(t_size[:30, 1])
    size_h_max = max(t_size[869:, 3])
    size_w_max = max(t_size[869:, 2])

    total_h = x_450 + size_h_max
    total_w = y_450 + size_w_max

    Parallel(n_jobs=num_cores, require='sharedmem')(
        delayed(save_panorama_frame_less_than_450)(i) for i in range(450))

    Parallel(n_jobs=num_cores, require='sharedmem')(
        delayed(save_panorama_frame_more_than_450)(i) for i in range(450, 900))

    create_video("res05_reference_plane.mp4")

    print(time.time() - s)