import cv2
import numpy as np


sobel_kernel_x = np.asarray([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_kernel_y = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def compute_gradient(image, kernel):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0.5)
    image_gradient = cv2.filter2D(blurred_image, cv2.CV_64F, kernel)
    image_gradient = np.abs(image_gradient)
    (b, g, r) = cv2.split(image_gradient)
    max_gradient = np.maximum(np.maximum(r, g), b)
    return max_gradient


def non_maximum_suppression(image, window_size):
    img_copy = np.copy(image)
    r = window_size // 2
    corner_indices = np.where(img_copy > 0)
    corner_x_indices = corner_indices[0]
    corner_y_indices = corner_indices[1]
    for i in range(corner_x_indices.shape[0]):
        x, y = corner_x_indices[i], corner_y_indices[i]
        window = img_copy[x - r: x + r + 1, y - r: y + r + 1]
        max_value_in_window = window.max()
        if img_copy[x, y] < max_value_in_window:
            img_copy[x, y] = 0
    return img_copy


def visualize_corners(image, corner_mask):
    img_copy = np.copy(image)
    corner_indices = np.where(corner_mask > 0)
    corner_x_indices = corner_indices[0]
    corner_y_indices = corner_indices[1]
    for i in range(corner_x_indices.shape[0]):
        cv2.circle(img_copy, (corner_y_indices[i], corner_x_indices[i]), 1, (0,0,255), -1)
    return img_copy


def detect_harris_corners(image, ksize, sigma, k, window_size):
    Ix = compute_gradient(cv2.GaussianBlur(image, (5, 5), 1), sobel_kernel_x)
    Iy = compute_gradient(cv2.GaussianBlur(image, (5, 5), 1), sobel_kernel_y)
    Ix_2 = Ix * Ix
    Iy_2 = Iy * Iy
    IxIy = Ix * Iy
    _, gradiant_direction = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)
    gradiant_magnitude = np.sqrt(Ix_2 + Iy_2)
    Sx_2 = cv2.GaussianBlur(Ix_2, (ksize, ksize), sigma)
    Sy_2 = cv2.GaussianBlur(Iy_2, (ksize, ksize), sigma)
    SxSy = cv2.GaussianBlur(IxIy, (ksize, ksize), sigma)
    det = Sx_2 * Sy_2 - SxSy * SxSy
    trace = Sx_2 + Sy_2
    R = det - k * (trace * trace)
    threshold = 0.01 * R.max()
    thresh = np.copy(R)
    thresh[R < threshold] = 0
    corner_mask = non_maximum_suppression(thresh, window_size)
    image_with_detected_corners = visualize_corners(image, corner_mask)
    return gradiant_direction, gradiant_magnitude, R, thresh, corner_mask, image_with_detected_corners


def weighted_histogram(mag, dir):
    histogram = np.zeros(8)
    for i in range(4):
        for j in range(4):
            d = int(dir[i, j] / 45)
            histogram[d] += mag[i, j]
    return histogram


def sift_descriptor(gradient_magnitude, grad_dir, h, w, x, y):
    x_start, x_end = x - 8, x + 7
    y_start, y_end = y - 8, y + 7
    if x_start < 0 or y_start < 0 or x_end > h or y_end > w:
        return None
    # divide into 4 * 4 windows
    sift_descriptor = np.zeros(128)
    histogram_bin_index = 0
    for i in range(4):
        for j in range(4):
            magnitude = gradient_magnitude[x_start + 4 * i: x_start + 4 * (i + 1), y_start + 4 * j: y_start + 4 * (j + 1)]
            dir = grad_dir[x_start + 4 * i: x_start + 4 * (i + 1), y_start + 4 * j: y_start + 4 * (j + 1)]
            sift_descriptor[histogram_bin_index * 8: (histogram_bin_index + 1) * 8] = weighted_histogram(magnitude, dir)
            histogram_bin_index += 1
    return sift_descriptor


def extract_all_sift_descriptors(im, corner_mask, gradient_magnitude, grad_dir):
    h, w = im.shape[:2]
    corner_indices = np.where(corner_mask > 0)
    corner_x_indices = corner_indices[0]
    corner_y_indices = corner_indices[1]
    num_corners = corner_x_indices.shape[0]
    all_sift_descriptor = np.zeros((num_corners, 130))
    for i in range(num_corners):
        x, y = corner_x_indices[i], corner_y_indices[i]
        des = sift_descriptor(gradient_magnitude, grad_dir, h, w, x, y)
        if des is not None:
            all_sift_descriptor[i, 0] = x
            all_sift_descriptor[i, 1] = y
            all_sift_descriptor[i, 2:] = des
    return all_sift_descriptor


def l1_distance(v1, v2):
    return np.sum(np.abs(v1 - v2, 2))


def l2_distance(v1, v2):
    return np.sum(np.power(v1 - v2, 2))


def find_nearest_neighbors(feature_vector, other_all_sift_descriptor):
    distance_1, distance_2 = float('inf'), float('inf')
    point_1, point_2 = 0, 0
    for i in range(other_all_sift_descriptor.shape[0]):
        vector = other_all_sift_descriptor[i, 2:]
        distance = l2_distance(feature_vector, vector)
        if distance < distance_1:
            distance_2 = distance_1
            point_2 = point_1
            distance_1 = distance
            point_1 = other_all_sift_descriptor[i, :2]
        elif distance < distance_2:
            distance_2 = distance
            point_2 = other_all_sift_descriptor[i, :2]
    return point_1, point_2, distance_1, distance_2


def match_feature_points(im1, corner_mask1, grad_mag1, grad_dir1, im2, corner_mask2, grad_mag2, grad_dir2, n, threshold):
    all_sift_descriptor1 = extract_all_sift_descriptors(im1, corner_mask1, grad_mag1, grad_dir1)
    all_sift_descriptor2 = extract_all_sift_descriptors(im2, corner_mask2, grad_mag2, grad_dir2)
    matching_points_map = (-1) * np.ones_like(im1)
    for i in range(all_sift_descriptor1.shape[0]):
        x, y = all_sift_descriptor1[i, :2]
        v = all_sift_descriptor1[i, 2:]
        p1, p2, d1, d2 = find_nearest_neighbors(v, all_sift_descriptor2)
        if d1 / d2 < threshold:
            matching_points_map[int(x), int(y), :2] = p1
    matched_keypoints = []
    mask1 = np.zeros((im1.shape[0], im1.shape[1]))
    mask2 = np.zeros((im2.shape[0], im2.shape[1]))
    for i in range(all_sift_descriptor2.shape[0]):
        x, y = all_sift_descriptor2[i, :2]
        x, y = int(x), int(y)
        v = all_sift_descriptor2[i, 2:]
        p1, p2, d1, d2 = find_nearest_neighbors(v, all_sift_descriptor1)
        if d1 / d2 < threshold:
            x_p1, y_p1 = p1
            x_p1, y_p1 = int(x_p1), int(y_p1)
            a, b = matching_points_map[x_p1, y_p1, :2]
            a, b = int(a), int(b)
            if x_p1 >= 0 and y_p1 >= 0 and a == x and b == y:
                matched_keypoints.append([x_p1, y_p1, x, y])
                mask1[x_p1, y_p1] = 1
                mask2[x, y] = 1
    return visualize_corners(im1, mask1), visualize_corners(im2, mask2), matched_keypoints


def visualize_feature_matches(corres1, corres2, match_points, num_random=30):
    h1, w1 = corres1.shape[:2]
    h2, w2 = corres2.shape[:2]
    res = np.zeros((max(h1, h2), w1 + w2, 3))
    res[:h1, :w1, :] = corres1
    res[:h2, w1:, :] = corres2
    l = len(match_points)
    for _ in range(num_random):
        match = match_points[np.random.randint(l)]
        x1, y1, x2, y2 = match
        cv2.line(res, (y1, x1), (y2 + w1, x2), (0, 0, 255), lineType=cv2.LINE_AA)
    return res


def opencv_harris_corners(image, gray_image, image_name):
    dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    im_copy = np.copy(image)
    im_copy[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite(image_name, im_copy)


ksize = 5
sigma = 1
k = 0.04
window_size = 9
im1 = cv2.imread('im01.jpg')
grad_dir1, grad1, R1, thresh1, corner_mask1, harris1 = detect_harris_corners(im1, ksize, sigma, k, window_size)
cv2.imwrite("res01_grad.jpg", cv2.normalize(src=grad1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
cv2.imwrite("res03_score.jpg", cv2.normalize(src=R1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
cv2.imwrite("res05_thresh.jpg", cv2.normalize(src=thresh1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
cv2.imwrite("res07_harris.jpg", harris1)

im2 = cv2.imread('im02.jpg')
grad_dir2, grad2, R2, thresh2, corner_mask2, harris2 = detect_harris_corners(im2, ksize, sigma, k, window_size)
cv2.imwrite("res02_grad.jpg", cv2.normalize(src=grad2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
cv2.imwrite("res04_score.jpg", cv2.normalize(src=R2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
cv2.imwrite("res06_thresh.jpg", cv2.normalize(src=thresh2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
cv2.imwrite("res08_harris.jpg", harris2)

n = 16
threshold = 0.8
corres1, corres2, match_points = match_feature_points(im1, corner_mask1, grad1, grad_dir1, im2, corner_mask2, grad2, grad_dir2, n, threshold)
cv2.imwrite("res09_corres.jpg", corres1)
cv2.imwrite("res10_corres.jpg", corres2)

res11 = visualize_feature_matches(corres1, corres2, match_points)
cv2.imwrite("res11.jpg", res11)