import cv2
import numpy as np


def homography_matrix(kp1, kp2, match_list):
    aList = []
    for match in match_list:
        x1, y1 = kp1[match.queryIdx].pt
        x1, y1 = x1, y1
        x2, y2 = kp2[match.trainIdx].pt
        x2, y2 = x2, y2

        a1 = [-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2]
        a2 = [0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2]
        aList.append(a1)
        aList.append(a2)

    A = np.matrix(aList)
    u, s, v = np.linalg.svd(A)
    M = np.reshape(v[-1], (3, 3))
    M = M / M[2,2]
    return M


def distance(M, x1, y1, x2, y2):
    p1 = np.array([x1, y1, 1])
    p2 = np.array([x2, y2, 1])

    # estimated_p1 = np.dot(np.linalg.inv(M), p2)
    # estimated_p1 = estimated_p1 * (1 / estimated_p1[0, 2])
    # error1 = np.linalg.norm(np.transpose(p1) - estimated_p1)

    estimated_p2 = np.dot(M, np.transpose(p1))
    estimated_p2 = estimated_p2 / estimated_p2[0, 2]
    error2 = np.linalg.norm(np.transpose(p2) - estimated_p2)
    return error2


def ransac(matches, kp1, kp2, N=1000, distance_threshold=50, inlier_threshold=0.25):
    l = len(matches)
    best_inlier_mask = np.zeros(l)
    best_inliers = []
    for i in range(N):
        sample_indices = np.random.choice(len(matches), 4, replace=False)
        sample = [matches[i] for i in sample_indices]
        M = homography_matrix(kp1, kp2, sample)
        current_inlier_mask = np.zeros(l)
        current_inliers = []
        match_index = 0
        for match in matches:
            x1, y1 = kp1[match.queryIdx].pt
            x1, y1 = int(x1), int(y1)
            x2, y2 = kp2[match.trainIdx].pt
            x2, y2 = int(x2), int(y2)
            d = distance(M, x1, y1, x2, y2)
            if d < distance_threshold:
                current_inlier_mask[match_index] = 1
                current_inliers.append(match)
            match_index += 1

        max_inlier_num = np.sum(best_inlier_mask)
        current_inlier_num = np.sum(current_inlier_mask)
        if current_inlier_num > max_inlier_num:
            best_inlier_mask = np.copy(current_inlier_mask)
            best_inliers = current_inliers.copy()
        if max_inlier_num > (l * inlier_threshold):
            print("here")
            break
    final_M = homography_matrix(kp1, kp2, best_inliers)
    return final_M, best_inlier_mask


def draw_keypoints(image, keypoint_list):
    copy_image = np.copy(image)
    for keypoint in keypoint_list:
        x, y = keypoint.pt
        x, y = int(x), int(y)
        cv2.circle(copy_image, (x, y), 1, (0, 255, 0), -1)
    return copy_image


def draw_correspondences(im1, im2, kp1, kp2, matches):
    copy_im1 = np.copy(im1)
    copy_im2 = np.copy(im2)
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    for match in matches:
        x1, y1 = kp1[match.queryIdx].pt
        x1, y1 = int(x1), int(y1)
        cv2.circle(copy_im1, (x1, y1), 2, (255, 0, 0), -1)
        x2, y2 = kp2[match.trainIdx].pt
        x2, y2 = int(x2), int(y2)
        cv2.circle(copy_im2, (x2, y2), 2, (255, 0, 0), -1)
    correspondences = np.zeros((max(h1, h2), w1 + w2, 3))
    correspondences[:h1, :w1, :] = copy_im1
    correspondences[:h2, w1:, :] = copy_im2
    return correspondences


def draw_match_lines(image, w1, matches, kp1, kp2):
    copy_image = np.copy(image)
    for match in matches:
        x1, y1 = kp1[match.queryIdx].pt
        x1, y1 = int(x1), int(y1)
        x2, y2 = kp2[match.trainIdx].pt
        x2, y2 = int(x2), int(y2)
        cv2.line(copy_image, (x1, y1), (x2 + w1, y2), (255, 0, 0), lineType=cv2.LINE_AA)
    return copy_image


def draw_random_match_lines(image, w1, matches, kp1, kp2, num_matches=20):
    copy_image = np.copy(image)
    l = len(matches)
    for _ in range(num_matches):
        r = np.random.randint(0, l)
        match = matches[r]
        x1, y1 = kp1[match.queryIdx].pt
        x1, y1 = int(x1), int(y1)
        x2, y2 = kp2[match.trainIdx].pt
        x2, y2 = int(x2), int(y2)
        cv2.line(copy_image, (x1, y1), (x2 + w1, y2), (255, 0, 0), lineType=cv2.LINE_AA)
    return copy_image


def draw_inliers(match_image, matches, inlier_mask, im1, im2, kp1, kp2):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    inlier_image = np.copy(match_image)
    mismatch_image = np.zeros((max(h1, h2), w1 + w2, 3))
    mismatch_image[:h1, :w1, :] = im1
    mismatch_image[:h2, w1:, :] = im2
    indices = np.where(inlier_mask > 0)[0]
    for index in indices:
        match = matches[index]
        x1, y1 = kp1[match.queryIdx].pt
        x1, y1 = int(x1), int(y1)
        x2, y2 = kp2[match.trainIdx].pt
        x2, y2 = int(x2), int(y2)
        cv2.circle(inlier_image, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(inlier_image, (x2 + w1, y2), 3, (0, 0, 255), -1)
        cv2.line(inlier_image, (x1, y1), (x2 + w1, y2), (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.line(mismatch_image, (x1, y1), (x2 + w1, y2), (0, 0, 255), lineType=cv2.LINE_AA)
    return inlier_image, mismatch_image


def warp_image_with_homography(M, im1, im2):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, M)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])

    res20 = cv2.warpPerspective(im2, Ht.dot(M), (xmax-xmin, ymax-ymin))
    res20_rect = np.copy(res20)
    warped_h = res20_rect.shape[0]
    rect_pts = np.array([[t[0], t[1]], [w1+t[0], t[1]], [w1+t[0], h1+t[1]], [t[0], h1+t[1]]], np.int32)
    rect_pts = rect_pts.reshape((-1, 1, 2))
    cv2.polylines(res20_rect,  [rect_pts], True, 255, 5, cv2.LINE_AA)
    res21 = np.zeros((max(h1, warped_h), w1 + res20_rect.shape[1], 3))
    res21[:h1, :w1, :] = im1
    res21[:warped_h, w1:, :] = res20_rect
    return res20, res21


im3 = cv2.imread('im03.jpg')
im4 = cv2.imread('im04.jpg')

h3, w3 = im3.shape[:2]
h4, w4 = im4.shape[:2]

im3_gray = cv2.imread('im03.jpg', cv2.IMREAD_GRAYSCALE)
im4_gray = cv2.imread('im04.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp3, des3 = sift.detectAndCompute(im3_gray, None)
kp4, _ = sift.detectAndCompute(im4_gray, None)

keypoint_im3 = draw_keypoints(im3, kp3)
keypoint_im4 = draw_keypoints(im4, kp4)
corners = np.zeros((h3, w3 + w4, 3))
corners[:h3, :w3, :] = keypoint_im3
corners[:h4, w3:, :] = keypoint_im4
cv2.imwrite("res22_corners.jpg", corners)

im4_gray = im4_gray[:, :int(w4//2)]
kp4, des4 = sift.detectAndCompute(im4_gray, None)


BFMatcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
matches = BFMatcher.match(queryDescriptors=des3, trainDescriptors=des4)
matches = sorted(matches, key=lambda x: x.distance)
correspondences = draw_correspondences(keypoint_im3, keypoint_im4, kp3, kp4, matches)
cv2.imwrite("res23_correspondences.jpg", correspondences)

match_image = draw_match_lines(correspondences, w3, matches, kp3, kp4)
cv2.imwrite("res24_matches.jpg", match_image)

random_match_image = draw_random_match_lines(correspondences, w3, matches, kp3, kp4)
cv2.imwrite("res25.jpg", random_match_image)


src_pts = np.float32([kp4[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
dst_pts = np.float32([kp3[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)

M, mask = ransac(matches, kp3, kp4)
M = np.linalg.inv(M)
inlier_mask = mask.ravel()
print(M)
print(np.sum(mask))
inlier_image, mismatch_image = draw_inliers(match_image, matches, inlier_mask, im3, im4, kp3, kp4)
cv2.imwrite("res26.jpg", inlier_image)
cv2.imwrite("res27_mismatch.jpg", mismatch_image)

pts = np.float32([[0, 0], [0, h3 - 1], [w3 - 1, h3 - 1], [w3 - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, np.linalg.inv(M))
im4_rect = np.copy(im4)
im4_rect = cv2.polylines(im4_rect, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
res19 = np.zeros((h3, w3 + w4, 3))
res19[:h3, :w3, :] = im3
res19[:h4, w3:, :] = im4_rect
cv2.imwrite("res28.jpg", res19)

res20, res21 = warp_image_with_homography(M, im3, im4)
cv2.imwrite("res29.jpg", res20)
cv2.imwrite("res30.jpg", res21)