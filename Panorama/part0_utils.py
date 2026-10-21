from pickle import dump, load
import cv2
import numpy as np
import multiprocessing
from joblib import Parallel, delayed


def extract_video_frames(video_name):
    frames = []
    vidCap = cv2.VideoCapture(video_name)
    vidCap.set(cv2.CAP_PROP_POS_MSEC, 0)
    frame_idx = 0
    while True:
        hasFarame, frame = vidCap.read()
        if hasFarame:
            cv2.imwrite("Frames/image_" + str(frame_idx) + ".jpg", frame)
            frame_idx += 1
        else:
            break
        if frame_idx >= 900:
            break
    return frames


def compute_homography_between_frames(kp1, des1, kp2, des2):
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def calculate_warped_bounds(H):
    warped_pts = cv2.perspectiveTransform(pts, H)
    concatenate_pts = np.concatenate((pts, warped_pts), axis=0)
    [xmin, ymin] = np.int32(concatenate_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(concatenate_pts.max(axis=0).ravel() + 0.5)
    out_size = (xmax - xmin, ymax - ymin)
    translation = [-xmin, -ymin]
    return translation, out_size


def apply_homography(pts, H, frame):
    warped_pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = np.int32(warped_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(warped_pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    warped = cv2.warpPerspective(frame, Ht.dot(H), (xmax - xmin, ymax - ymin))
    return warped


def save_homography(H, filename):
    with open(filename, "wb") as f:
        dump(H, f)


def load_homography(filename):
    with open(filename, "rb") as f:
        H = load(f)
    return H


def save_homography_less_than_90(i):
    # print(i)
    frame_i = cv2.imread("Frames/image_" + str(i) + ".jpg")
    kp_i, des_i = sift.detectAndCompute(frame_i, None)
    H_i = compute_homography_between_frames(kp90, des90, kp_i, des_i).dot(H_90)
    t_i, size_i = calculate_warped_bounds(H_i)
    t_size[i, :] = [t_i[0], t_i[1], size_i[0], size_i[1]]
    save_homography(H_i, "Homography/H_" + str(i) + ".h")
    warped_i = apply_homography(pts, H_i, frame_i)
    cv2.imwrite("WarpedFrames/w_" + str(i) + ".jpg", warped_i)


def save_homography_less_than_270(i):
    # print(i)
    frame_i = cv2.imread("Frames/image_" + str(i) + ".jpg")
    kp_i, des_i = sift.detectAndCompute(frame_i, None)
    H_i = compute_homography_between_frames(kp270, des270, kp_i, des_i).dot(H_270)
    t_i, size_i = calculate_warped_bounds(H_i)
    t_size[i, :] = [t_i[0], t_i[1], size_i[0], size_i[1]]
    save_homography(H_i, "Homography/H_" + str(i) + ".h")
    warped_i = apply_homography(pts, H_i, frame_i)
    cv2.imwrite("WarpedFrames/w_" + str(i) + ".jpg", warped_i)


def save_homography_near_450(i):
    # print(i)
    frame_i = cv2.imread("Frames/image_" + str(i) + ".jpg")
    kp_i, des_i = sift.detectAndCompute(frame_i, None)
    H_i = compute_homography_between_frames(kp450, des450, kp_i, des_i)
    t_i, size_i = calculate_warped_bounds(H_i)
    t_size[i, :] = [t_i[0], t_i[1], size_i[0], size_i[1]]
    save_homography(H_i, "Homography/H_" + str(i) + ".h")
    warped_i = apply_homography(pts, H_i, frame_i)
    cv2.imwrite("WarpedFrames/w_" + str(i) + ".jpg", warped_i)


def save_homography_more_than_630(i):
    # print(i)
    frame_i = cv2.imread("Frames/image_" + str(i) + ".jpg")
    kp_i, des_i = sift.detectAndCompute(frame_i, None)
    H_i = compute_homography_between_frames(kp630, des630, kp_i, des_i).dot(H_630)
    t_i, size_i = calculate_warped_bounds(H_i)
    t_size[i, :] = [t_i[0], t_i[1], size_i[0], size_i[1]]
    save_homography(H_i, "Homography/H_" + str(i) + ".h")
    warped_i = apply_homography(pts, H_i, frame_i)
    cv2.imwrite("WarpedFrames/w_" + str(i) + ".jpg", warped_i)


def save_homography_more_than_810(i):
    # print(i)
    frame_i = cv2.imread("Frames/image_" + str(i) + ".jpg")
    kp_i, des_i = sift.detectAndCompute(frame_i, None)
    H_i = compute_homography_between_frames(kp810, des810, kp_i, des_i).dot(H_810)
    t_i, size_i = calculate_warped_bounds(H_i)
    t_size[i, :] = [t_i[0], t_i[1], size_i[0], size_i[1]]
    save_homography(H_i, "Homography/H_" + str(i) + ".h")
    warped_i = apply_homography(pts, H_i, frame_i)
    cv2.imwrite("WarpedFrames/w_" + str(i) + ".jpg", warped_i)


def save_warped_frames_with_homography(i):
    # print(i)
    frame_i = cv2.imread("Frames/image_" + str(i) + ".jpg")
    H_i = load_homography("Homography/H_" + str(i) + ".h")
    t_i, size_i = calculate_warped_bounds(H_i)
    t_size[i, :] = [t_i[0], t_i[1], size_i[0], size_i[1]]
    warped_i = apply_homography(pts, H_i, frame_i)
    cv2.imwrite("WarpedFrames/w_" + str(i) + ".jpg", warped_i)


if __name__ == "__main__":
    extract_video_frames("video.mp4")
    num_cores = multiprocessing.cpu_count()

    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    frame_90 = cv2.imread("Frames/image_89.jpg")
    frame_270 = cv2.imread("Frames/image_269.jpg")
    frame_450 = cv2.imread("Frames/image_449.jpg")
    frame_630 = cv2.imread("Frames/image_629.jpg")
    frame_810 = cv2.imread("Frames/image_809.jpg")

    h, w = frame_450.shape[:2]
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

    t_size = np.zeros((900, 4))

    kp90, des90 = sift.detectAndCompute(frame_90, None)
    kp270, des270 = sift.detectAndCompute(frame_270, None)
    kp450, des450 = sift.detectAndCompute(frame_450, None)
    kp630, des630 = sift.detectAndCompute(frame_630, None)
    kp810, des810 = sift.detectAndCompute(frame_810, None)

    H_270 = compute_homography_between_frames(kp450, des450, kp270, des270)
    H_630 = compute_homography_between_frames(kp450, des450, kp630, des630)
    H_90 = compute_homography_between_frames(kp270, des270, kp90, des90).dot(H_270)
    H_810 = compute_homography_between_frames(kp630, des630, kp810, des810).dot(H_630)

    save_homography(H_90, "Homography/H_89.h")
    save_homography(H_270, "Homography/H_269.h")
    save_homography(np.identity(3), "Homography/H_449.h")
    save_homography(H_630, "Homography/H_629.h")
    save_homography(H_810, "Homography/H_809.h")

    t_90, size_90 = calculate_warped_bounds(H_90)
    t_size[89, :] = [t_90[0], t_90[1], size_90[0], size_90[1]]
    t_270, size_270 = calculate_warped_bounds(H_270)
    t_size[269, :] = [t_270[0], t_270[1], size_270[0], size_270[1]]
    t_size[449, :] = [0, 0, h, w]
    t_630, size_630 = calculate_warped_bounds(H_630)
    t_size[629, :] = [t_630[0], t_630[1], size_630[0], size_630[1]]
    t_810, size_810 = calculate_warped_bounds(H_810)
    t_size[809, :] = [t_810[0], t_810[1], size_810[0], size_810[1]]

    warped_90 = apply_homography(pts, H_90, frame_90)
    warped_270 = apply_homography(pts, H_270, frame_270)
    warped_630 = apply_homography(pts, H_630, frame_630)
    warped_810 = apply_homography(pts, H_810, frame_810)

    Parallel(n_jobs=num_cores, require='sharedmem')(
        delayed(save_homography_less_than_90)(i) for i in range(89))
    Parallel(n_jobs=num_cores, require='sharedmem')(
        delayed(save_homography_less_than_270)(i) for i in range(90, 269))
    Parallel(n_jobs=num_cores, require='sharedmem')(
        delayed(save_homography_near_450)(i) for i in range(270, 449))
    Parallel(n_jobs=num_cores, require='sharedmem')(
        delayed(save_homography_near_450)(i) for i in range(450, 629))
    Parallel(n_jobs=num_cores, require='sharedmem')(
        delayed(save_homography_more_than_630)(i) for i in range(630, 809))
    Parallel(n_jobs=num_cores, require='sharedmem')(
        delayed(save_homography_more_than_810)(i) for i in range(810, 900))


    # Parallel(n_jobs=num_cores, require='sharedmem')(
    #     delayed(save_warped_frames_with_homography)(i) for i in range(900))

    np.savetxt("t_size.txt", t_size, fmt='%d')


