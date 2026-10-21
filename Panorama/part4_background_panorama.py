import time
import cv2
import numpy as np
from pickle import dump, load


def get_clicked_point(event, x, y, flags, params):
    global points_image, clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([int(y), int(x)])
        cv2.circle(points_image, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('image', points_image)


def get_points(image):
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', get_clicked_point)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def euclidean_distance_from_point(h, w, x0, y0):
    res = np.zeros((h, w, 3))
    one = np.ones((h, w))
    x = np.arange(start=0, stop=h, step=1, dtype=float)
    y = np.arange(start=0, stop=w, step=1, dtype=float)
    yy, xx = np.meshgrid(y, x)
    d = np.abs(xx - (x0) * one) + np.abs(yy - (y0) * one)
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


def gaussian_stack(level, image, kernel_size, sigma):
    g_stack = []
    prev = image
    for l in range(level):
        level_sigma = np.power(sigma, l)
        if l > 0:
            prev = g_stack[l - 1]
        g_stack.append(cv2.GaussianBlur(prev, ksize=(kernel_size, kernel_size), sigmaX=level_sigma, sigmaY=level_sigma))
    # del g_stack[0]
    return g_stack


def laplacian_stack(level, image, g_stack):
    copy = cv2.GaussianBlur(image, ksize=(1, 1), sigmaX=0, sigmaY=0)
    l_stack = [image.astype('int') - g_stack[0].astype('int')]
    for l in range(0, level - 2):
        l_stack.append(g_stack[l].astype('int') - g_stack[l + 1].astype('int'))
    l_stack.append(g_stack[level - 1])
    return l_stack


def stack_blend(level, im1, im2, mask, kernel_size=45, sigma=1.2):
    h, w = im1.shape[:2]
    mask_stack = gaussian_stack(level, mask, kernel_size=kernel_size, sigma=2 * sigma)

    im1_g_stack = gaussian_stack(level, im1, kernel_size, sigma)
    im1_l_stack = laplacian_stack(level, im1, im1_g_stack)

    im2_g_stack = gaussian_stack(level, im2, kernel_size, sigma)
    im2_l_stack = laplacian_stack(level, im2, im2_g_stack)

    res = np.zeros((h, w, 3)).astype('uint8')
    blended_stack = []
    for l in range(level):
        im1_l = im1_l_stack[l]
        im2_l = im2_l_stack[l]
        mask1 = mask_stack[l]
        mask2 = np.ones((h, w, 3)) - mask1
        blended = (np.multiply(im1_l, mask1) + np.multiply(im2_l, mask2)).astype('uint8')
        blended_stack.append(blended)
        res += blended
    return res


def apply_homography(H, frame, pts):
    warped_pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = np.int32(warped_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(warped_pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    warped = cv2.warpPerspective(frame, Ht.dot(H), (xmax - xmin, ymax - ymin))
    return warped


def get_mask_list():
    list = []
    for i in indices:
        warped_i = cv2.imread("WarpedFrames/w_" + str(i) + ".jpg")
        mask_i = np.zeros_like(warped_i)
        if i == 0:
            point_i = point_list[0]
        elif i == 899:
            point_i = point_list[-1]
        else:
            point_i = point_list[(i + 1) // 90]
        cv2.line(mask_i, (point_i[1], point_i[0]), (point_i[3], point_i[2]), color=(1, 1, 1), thickness=2)
        X, Y, _ = np.where(mask_i == 1)
        for j in range(X.shape[0]):
            mask_i[:X[j], Y[j], :] = [1, 1, 1]
        list.append(mask_i)
        cv2.imwrite("mask_" + str(i) + ".jpg", mask_i * 255)
    return list


def background_of_frame(i):
    # print("i = " + str(i))
    warped_i = cv2.imread("WarpedFrames/w_" + str(i) + ".jpg")
    x_i, y_i = get_coords(i)
    h_i, w_i = warped_i.shape[:2]
    frames = []
    start = max(0, i - 45)
    stop = min(900, i + 45)
    if i == 449:
        start = max(0, i - 60)
        stop = min(900, i + 60)
    for j in range(start, stop):
        # print(j)
        p_j = cv2.imread("PanoromaFrames/p_" + str(j) + ".jpg")
        p_j = p_j[x_i:x_i + h_i, y_i:y_i + w_i, :]
        frames.append(p_j)
    if i == 0:
        mask_i = mask_list[0]
    elif i == 899:
        mask_i = mask_list[-1]
    else:
        mask_i = cv2.imread("mask_" + str(i) + ".jpg") // 255
    frame_median = np.median(frames, axis=0).astype(dtype=np.uint8)
    b_i = np.zeros_like(p_0)
    b_i[x_i:x_i + h_i, y_i:y_i + w_i, :] = mask_i * warped_i + (1 - mask_i) * frame_median
    cv2.imwrite("b_" + str(i + 1) + ".jpg", b_i)


def get_coords(i):
    if i < 449:
        return x_450 - t_size[i, 1], y_450 - t_size[i, 0]
    elif i == 449:
        return x_450, y_450
    else:
        warped_i = cv2.imread("WarpedFrames/w_" + str(i) + ".jpg")
        return x_450 + t_size[i, 3] - warped_i.shape[0], y_450 + t_size[i, 2] - warped_i.shape[1]


def create_panorama_background():
    panorama = np.zeros((p_h, p_w, 3))
    distance = np.zeros((p_h, p_w, 3))

    for i in indices:
        b_i = cv2.imread("b_" + str(i + 1) + ".jpg")
        H_i = load_homography("Homography/H_" + str(i) + ".h")
        d_i = apply_homography(H_i, c_distance, pts)
        if i == 449:
            d_i *= cv2.normalize(src=e_distance ** 2, dst=None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)
        if i == 629:
            d_i[:d_i.shape[0]//2, (2 * d_i.shape[1])//3:, :] *= 20
        if i > 630:
            d_i[:d_i.shape[0] // 2, :(2 * d_i.shape[1])//3, :] *= ((1 / 30) * ((450 - (i - 449))//90) ** 2)
            d_i[d_i.shape[0] // 2:, :(2 * d_i.shape[1])//3, :] *= ((1 / 5) * ((450 - (i - 449))//90) ** 2)
        x_i, y_i = get_coords(i)
        sum_warped = panorama[x_i:x_i + d_i.shape[0], y_i:y_i + d_i.shape[1], :] * distance[x_i:x_i + d_i.shape[0], y_i:y_i + d_i.shape[1], :] + b_i[x_i:x_i + d_i.shape[0], y_i:y_i + d_i.shape[1], :] * d_i
        sum_d = distance[x_i:x_i + d_i.shape[0], y_i:y_i + d_i.shape[1], :] + d_i
        panorama[x_i:x_i + d_i.shape[0], y_i:y_i + d_i.shape[1], :] = np.divide(sum_warped, sum_d, out=np.zeros_like(sum_warped), where=sum_d!=0)
        distance[x_i:x_i + d_i.shape[0], y_i:y_i + d_i.shape[1], :] += d_i
    cv2.imwrite("res06_background_panorama.jpg", panorama)


frame_450 = cv2.imread("Frames/image_449.jpg")
h, w = frame_450.shape[:2]
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
t_size = np.loadtxt("t_size.txt", dtype=int)
y_450 = max(t_size[:30, 0])
x_450 = max(t_size[:30, 1])
p_0 = cv2.imread("PanoromaFrames/p_0.jpg")
p_h, p_w = p_0.shape[:2]

indices = [0, 89, 179, 269, 359, 449, 539, 629, 719, 809, 899]

point_list = []
# point_list.append([1190, 0, 774, 2815])     # 0
point_list.append([0, 0, 0, 2815])          # 0
point_list.append([1008, 0, 630, 2492])     # 89
point_list.append([882, 0, 494, 2254])      # 179
point_list.append([766, 0, 374, 2103])      # 269
point_list.append([662, 0, 318, 1980])      # 359
point_list.append([542, 0, 0, 1920])        # 449
point_list.append([690, 1993, 0, 700])      # 539
point_list.append([794, 2128, 0, 250])      # 629
point_list.append([950, 2441, 70, 0])       # 719
point_list.append([1220, 2964, 220, 0])     # 809
point_list.append([1450, 3348, 300, 0])     # 899
mask_list = get_mask_list()

for i in indices:
    background_of_frame(i)

c_distance = chebyshev_distance(h, w)
e_distance = euclidean_distance(h, w)

create_panorama_background()