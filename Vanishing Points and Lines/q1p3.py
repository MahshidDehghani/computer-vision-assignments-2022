import cv2
import numpy as np


image = cv2.imread("vns.jpg")


f = 13780.280928096205
principal_point = np.array([1148.3085075, 932.65346809])

R = np.array([[-0.86334836, 0.50375015, 0.02941763],
 [0.09348737, 0.10238938, 0.99034162],
 [0.49587269, 0.85775999, -0.13549199]])


K = np.array([
    [f, 0, principal_point[0]],
    [0, f, principal_point[1]],
    [0, 0, 1]
])


# Rectification homography
H = K @ R.T @ np.linalg.inv(K)
H /= H[2, 2]

print("H =")
print(H)


# Warp image
height, width = image.shape[:2]

corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]]).T
warped_corners = H @ corners
warped_corners /= warped_corners[2, :]

xs, ys = warped_corners[0, :], warped_corners[1, :]
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()

T = np.array([[1, 0, -x_min],
              [0, 1, -y_min],
              [0, 0, 1]])

H_final = T @ H
out_w, out_h = int(x_max - x_min), int(y_max - y_min)

rectified_image = cv2.warpPerspective(image, H_final, (out_w, out_h))
cv2.imwrite("res04.jpg", rectified_image)