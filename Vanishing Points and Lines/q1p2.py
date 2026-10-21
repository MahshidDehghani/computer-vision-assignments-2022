import cv2
import numpy as np
from matplotlib import pyplot as plt


def calibration_matrix(v1, v2, v3):
    a1, b1 = v1
    a2, b2 = v2
    a3, b3 = v3
    system_matrix = np.array([[(a1 - a3), (b1 - b3)],
                  [(a2 - a3), (b2 - b3)]])
    rhs = np.array([a2 * (a1 - a3) + b2 * (b1 - b3), a1 * (a2 - a3) + b1 * (b2 - b3)])
    principal_point = np.linalg.solve(system_matrix, rhs)
    principal_x, principal_y = principal_point
    focal_length_squared = -principal_x ** 2 - principal_y ** 2 + (a1 + a2) * principal_x + (b1 + b2) * principal_y - (a1 * a2 + b1 * b2)
    f = np.sqrt(focal_length_squared)
    camera_matrix = np.array([[f, 0, principal_x],
                  [0, f, principal_y],
                  [0, 0, 1]])
    return principal_point, f, camera_matrix


def draw_principal_point():
    copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    principal_x, principal_y = principal_point
    principal_x, principal_y = int(principal_x), int(principal_y)
    cv2.circle(copy, (principal_x, principal_y), 30, (0, 255, 0), -1)
    plt.imshow(copy)
    title = "focal length: " + str(f)
    plt.title(title)
    plt.savefig("res03.jpg")


def calculate_Ri(camera_matrix_inv, Vi):
    v = np.array([[Vi[0]], [Vi[1]], [1]])
    kv = camera_matrix_inv.dot(v)
    lambda_i = 1 / np.linalg.norm(kv)
    R_i = lambda_i * kv
    return R_i.reshape(-1, 1)


def compute_rotation_matrix():
    R = np.zeros((3, 3))
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    rx = calculate_Ri(camera_matrix_inv, V_x)
    R[:, 0] = rx[:, 0]
    ry = calculate_Ri(camera_matrix_inv, V_y)
    R[:, 1] = ry[:, 0]
    rz = calculate_Ri(camera_matrix_inv, V_z)
    R[:, 2] = rz[:, 0]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R


image = cv2.imread("vns.jpg")
V_x = np.array([-22844.105631975086, 3530.663638723073])
V_y = np.array([9241.270066238067, 2577.582700175336])
V_z = np.array([-1843.626452189456, -99790.54155495981])
principal_point, f, camera_matrix = calibration_matrix(V_x, V_y, V_z)
print("P = \n", principal_point)
print("f = \n", f)
R = compute_rotation_matrix()
print("R = \n", R)
draw_principal_point()

