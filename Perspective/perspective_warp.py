import cv2
import numpy as np


def warp_image_using_homography(M, im):
    # Define corners of the image
    h, w, _ = im.shape
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]]).reshape(-1, 1, 2)

    # Calculate the transformed corners
    transformed_corners = cv2.perspectiveTransform(corners, M)

    # Get the bounding box of the transformed corners
    [xmin, ymin] = np.int32(transformed_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(transformed_corners.max(axis=0).ravel())

    # Translation matrix to ensure no negative coordinates
    translation = [-xmin, -ymin]
    translation_matrix = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    # Apply the homography and translation to the image
    result = cv2.warpPerspective(im, translation_matrix.dot(M), (xmax - xmin, ymax - ymin))
    return result


logo = cv2.imread("logo.png")
h, w, _ = logo.shape

# Camera intrinsics (for both cameras, assuming same focal length)
focal_length = 500
cx, cy = w / 2, h / 2  # Optical center
k = np.asarray([[focal_length, 0, cy],
                [0, focal_length, cx],
                [0, 0, 1]])

# Camera 1 (Side Camera) - Position: (40, 0, 25), Look at: (0, 0, 0)
theta = np.arctan2(-40, 25)
c, s = np.cos(theta), np.sin(theta)
# degree = theta * 180/np.pi
R1 = np.asarray([[1, 0, 0],
                [0, c, -s],
                [0, s, c]])
C1 = np.asarray([[40],
                [0],
                [25]])
t1 = -np.dot(R1, C1)
Rt1 = np.zeros((3, 4))
Rt1[:, :3] = R1
Rt1[:, 3] = t1[:, 0]

P1 = np.dot(k, Rt1)

# Camera 2 (Top-down Camera) - Position: (0, 0, 25), Looking down at the center
Rt2 = [[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, -25]]
P2 = np.dot(k, Rt2)

# Compute the homography H from Camera 2 to Camera 1
H = np.dot(P1[:, :3], np.linalg.inv(P2[:, :3]))  # Only consider rotation and translation part (3x3)

res = warp_image_using_homography(H, logo)
cv2.imwrite("res12.jpg", res)

