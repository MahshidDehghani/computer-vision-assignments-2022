import cv2
import numpy as np
from pickle import dump, load


def load_homography(filename):
    with open(filename, "rb") as f:
        H = load(f)
    return H


def euclidean_distance(h, w):
    res = np.zeros((h, w, 3))
    one = np.ones((h, w))
    x = np.arange(start=0, stop=h, step=1, dtype=float)
    y = np.arange(start=0, stop=w, step=1, dtype=float)
    yy, xx = np.meshgrid(y, x)
    d = np.abs(xx - (h/2) * one) + np.abs(yy - (w/2) * one)
    d = d.max() * one - d
    for i in range(3):
        res[:, :, i] = d
    return res


def chebyshev_distance(h, w):
    res = np.zeros((h, w, 3))
    one = np.ones((h, w))
    x = np.arange(start=0, stop=h, step=1, dtype=float)
    y = np.arange(start=0, stop=w, step=1, dtype=float)
    yy, xx = np.meshgrid(y, x)
    d = np.maximum(np.abs(xx - (h/2) * one), np.abs(yy - (w/2) * one))
    d = d.max() * one - d
    for i in range(3):
        res[:, :, i] = d
    return res


def apply_homography(H, frame, pts):
    warped_pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = np.int32(warped_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(warped_pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    warped = cv2.warpPerspective(frame, Ht.dot(H), (xmax - xmin, ymax - ymin))
    return warped


def euclidean_distance_blend():
    frame_90 = cv2.imread("Frames/image_89.jpg")
    frame_270 = cv2.imread("Frames/image_269.jpg")
    frame_450 = cv2.imread("Frames/image_449.jpg")
    frame_630 = cv2.imread("Frames/image_629.jpg")
    frame_810 = cv2.imread("Frames/image_809.jpg")

    H_90 = load_homography("Homography/H_89.h")
    H_270 = load_homography("Homography/H_269.h")
    H_630 = load_homography("Homography/H_629.h")
    H_810 = load_homography("Homography/H_809.h")

    warped_90 = cv2.imread("WarpedFrames/w_89.jpg")
    warped_270 = cv2.imread("WarpedFrames/w_269.jpg")
    warped_630 = cv2.imread("WarpedFrames/w_629.jpg")
    warped_810 = cv2.imread("WarpedFrames/w_809.jpg")

    h, w = frame_450.shape[:2]
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    t_size = np.loadtxt("t_size.txt", dtype=int)

    c_distance = chebyshev_distance(h, w)
    e_distance = euclidean_distance(h, w)

    d_90 = apply_homography(H_90, c_distance, pts)
    cv2.imwrite("d_90.jpg", cv2.normalize(src=d_90, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    d_270 = apply_homography(H_270, c_distance, pts)
    cv2.imwrite("d_270.jpg", cv2.normalize(src=d_270, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    d_450 = c_distance * cv2.normalize(src=e_distance ** 2, dst=None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite("d_450.jpg", cv2.normalize(src=d_450, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    d_630 = apply_homography(H_630, c_distance, pts)
    cv2.imwrite("d_630.jpg", cv2.normalize(src=d_630, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    d_810 = apply_homography(H_810, c_distance, pts)


    warped_630_h, warped_630_w = warped_630.shape[:2]
    warped_810_h, warped_810_w = warped_810.shape[:2]

    total_h = t_size[89, 1] + t_size[809, 3]
    total_w = max(t_size[89, 0], t_size[809, 0]) + max(t_size[89, 2] - t_size[89, 0], t_size[809, 2] - t_size[809, 0])
    x_90, y_90 = 0, max(0, t_size[809, 0] - t_size[89, 0])
    x_270, y_270 = (t_size[89, 1] - t_size[269, 1]), (t_size[89, 0] - t_size[269, 0])
    x_450, y_450 = t_size[89, 1], t_size[89, 0]
    x_630, y_630 = x_450 + t_size[629, 3] - warped_630_h, y_450 + t_size[629, 2] - warped_630_w
    x_810, y_810 = total_h - warped_810_h, y_450 + t_size[809, 2] - warped_810_w

    d_810[:warped_810_h//2, :y_630 + warped_630.shape[1] - y_810, :] *= (1/30)
    d_810[warped_810_h//2:, :y_630 + warped_630.shape[1] - y_810, :] *= (1/5)
    cv2.imwrite("d_810.jpg", cv2.normalize(src=d_810, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))

    panorama = np.zeros((total_h, total_w, 3))
    distance = np.zeros((total_h, total_w, 3))
    panorama[x_90:x_90 + warped_90.shape[0], y_90:y_90 + warped_90.shape[1], :] = warped_90
    distance[x_90:x_90 + warped_90.shape[0], y_90:y_90 + warped_90.shape[1], :] = d_90

    sum_warped = panorama[x_270:warped_270.shape[0] + x_270, y_270:y_270 + warped_270.shape[1], :] * distance[x_270:warped_270.shape[0] + x_270, y_270:y_270 + warped_270.shape[1], :] + warped_270 * d_270
    sum_d = distance[x_270:warped_270.shape[0] + x_270, y_270:y_270 + warped_270.shape[1], :] + d_270
    panorama[x_270:warped_270.shape[0] + x_270, y_270:y_270 + warped_270.shape[1], :] = np.divide(sum_warped, sum_d, out=np.zeros_like(sum_warped), where=sum_d!=0)
    distance[x_270:warped_270.shape[0] + x_270, y_270:y_270 + warped_270.shape[1], :] += d_270

    sum_warped = panorama[x_450:x_450 + frame_450.shape[0], y_450:y_450 + frame_450.shape[1], :] * distance[x_450:x_450 + frame_450.shape[0], y_450:y_450 + frame_450.shape[1], :] + frame_450 * d_450
    sum_d = distance[x_450:x_450 + frame_450.shape[0], y_450:y_450 + frame_450.shape[1], :] + d_450
    panorama[x_450:x_450 + frame_450.shape[0], y_450:y_450 + frame_450.shape[1], :] = np.divide(sum_warped, sum_d, out=np.zeros_like(sum_warped), where=sum_d!=0)
    distance[x_450:x_450 + frame_450.shape[0], y_450:y_450 + frame_450.shape[1], :] += d_450

    sum_warped = panorama[x_630:x_630 + warped_630.shape[0], y_630:y_630 + warped_630.shape[1], :] * distance[x_630:x_630 + warped_630.shape[0], y_630:y_630 + warped_630.shape[1], :] + warped_630 * d_630
    sum_d = distance[x_630:x_630 + warped_630.shape[0], y_630:y_630 + warped_630.shape[1], :] + d_630
    panorama[x_630:x_630 + warped_630.shape[0], y_630:y_630 + warped_630.shape[1], :] = np.divide(sum_warped, sum_d, out=np.zeros_like(sum_warped), where=sum_d!=0)
    distance[x_630:x_630 + warped_630.shape[0], y_630:y_630 + warped_630.shape[1], :] += d_630

    sum_warped = panorama[x_810:x_810 + warped_810.shape[0], y_810:y_810 + warped_810.shape[1], :] * distance[x_810:x_810 + warped_810.shape[0], y_810:y_810 + warped_810.shape[1], :] + warped_810 * d_810
    sum_d = distance[x_810:x_810 + warped_810.shape[0], y_810:y_810 + warped_810.shape[1], :] + d_810
    panorama[x_810:x_810 + warped_810.shape[0], y_810:y_810 + warped_810.shape[1], :] = np.divide(sum_warped, sum_d, out=np.zeros_like(sum_warped), where=sum_d!=0)
    distance[x_810:x_810 + warped_810.shape[0], y_810:y_810 + warped_810.shape[1], :] += d_810
    # cv2.imwrite("dist.jpg", cv2.normalize(src=distance, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    return panorama


def create_panorama_key_frames():
    panorama = euclidean_distance_blend()
    cv2.imwrite("res04_key_frames_panorama.jpg", panorama)


create_panorama_key_frames()
