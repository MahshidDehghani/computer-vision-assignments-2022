import math

import cv2
import numpy as np


def on_mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_image, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', display_image)


def get_clicked_points():
    cv2.imshow('image', display_image)
    cv2.setMouseCallback('image', on_mouse_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_edges(gray_image):
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    blurred_image[:, :int(w/5)] = np.zeros((h, int(w/5)))
    edge_image = cv2.Canny(blurred_image, 40, 255)
    return edge_image


def line_from_points(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    if x1 != x2:
        slope = (y2 - y1) / (x2 - x1)
    else:
        slope = 100000000
    intercept = y2 - slope * x2
    return [slope, intercept]


def draw_line(image, line, color):
    h, w = image.shape[:2]
    slope, intercept = line
    b = -1
    point_list = []
    y = 0
    x = int(-(y * slope + intercept) / b)
    if 0 <= x < h:
        point_list.append((y, x))
    y = w
    x = int(-(y * slope + intercept) / b)
    if 0 <= x < h:
        point_list.append((y, x))
    x = 0
    y = int(-(x * b + intercept) / slope)
    if 0 <= y < w:
        point_list.append((y, x))
    x = h
    y = int(-(x * b + intercept) / slope)
    if 0 <= y < w:
        point_list.append((y, x))
    cv2.line(image, point_list[0], point_list[1], color)
    return image


def draw_selected_lines():
    copy = np.copy(image)
    for i in range(int(len(clicked_points) / 2)):
        if i < 5:
            color = (0, 255, 255)        # yellow
        elif i < 10:
            color = (255, 255, 0)        # blue
        else:
            color = (0, 255, 0)          # green
        pt1, pt2 = clicked_points[2 * i], clicked_points[2 * i + 1]
        # line = get_line(pt1, pt2)
        # draw_line(image, line, color)
        cv2.line(copy, (pt1[0], pt1[1]), (pt2[0], pt2[1]), color, 2, cv2.LINE_AA)
    cv2.imwrite("selected_lines.jpg", copy)


def group_lines_by_direction():
    x_lines = []
    y_lines = []
    z_lines = []
    for i in range(int(len(clicked_points) / 2)):
        pt1, pt2 = clicked_points[2 * i], clicked_points[2 * i + 1]
        line = line_from_points(pt1, pt2)
        if i < 5:
            x_lines.append(line)
        elif i < 10:
            y_lines.append(line)
        else:
            z_lines.append(line)
    return x_lines, y_lines, z_lines


def point_to_line_distance(slope, intercept, pt_x, pt_y):
    om = (-1 / slope)
    oc = pt_y - om * pt_x
    x = (intercept - oc) / (om - slope)
    y = om * x + oc
    d = math.sqrt((y - pt_y) ** 2 + (x - pt_x) ** 2)
    return d


def vanishing_point_error(lines, v_x, v_y):
    sum_dist = 0
    for k in range(len(lines)):
        slope, intercept = lines[k]
        d = point_to_line_distance(slope, intercept, v_x, v_y)
        sum_dist += d ** 2
    sum_dist = math.sqrt(sum_dist)
    return sum_dist


def estimate_vanishing_point(lines):
    vanishing_point = None
    min_error = 100000000000
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            m1, c1 = lines[i]
            m2, c2 = lines[j]
            if m1 != m2:
                v_x = (c1 - c2) / (m2 - m1)
                v_y = m1 * v_x + c1
                error = vanishing_point_error(lines, v_x, v_y)
                if min_error > error:
                    min_error = error
                    vanishing_point = [v_x, v_y]
    return vanishing_point


def draw_horizon():
    res1 = np.copy(image)
    h, w = res1.shape[:2]
    a, b, c = horizon_line
    point_list = []
    y = 0
    x = int(-(y * a + c) / b)
    if 0 <= x < h:
        point_list.append((y, x))
    y = w
    x = int(-(y * a + c) / b)
    if 0 <= x < h:
        point_list.append((y, x))
    x = 0
    y = int(-(x * b + c) / a)
    if 0 <= y < w:
        point_list.append((y, x))
    x = h
    y = int(-(x * b + c) / a)
    if 0 <= y < w:
        point_list.append((y, x))
    cv2.line(res1, point_list[0], point_list[1], (0, 255, 0), 3)
    cv2.imwrite("res01.jpg", res1)


def draw_vanishing_points():
    n = 10
    y_start, x_start = min(Vx[0], Vy[0], Vz[0]), min(Vx[1], Vy[1], Vz[1])
    y_end, x_end = max(Vx[0], Vy[0], Vz[0]), max(Vx[1], Vy[1], Vz[1])
    y_start, x_start = int(y_start / n), int(x_start / n)
    y_end, x_end = int(y_end / n), int(x_end / n)
    res_h, res_w = (x_end - x_start), (y_end - y_start)
    res2 = np.zeros((res_h + 100, res_w + 100, 3))
    resized_h, resized_w = int(h/n), int(w/n)
    resized_image = cv2.resize(image, (resized_w, resized_h))
    res2[50 - x_start:50 - x_start + resized_h, 50 - y_start:50 - y_start + resized_w, :] = resized_image
    cv2.circle(res2, (50 + int(Vx[0]/n)-y_start, 50 + int(Vx[1]/n)-x_start), 20, (255, 255, 255), -1)
    cv2.circle(res2, (50 + int(Vy[0]/n)-y_start, 50 + int(Vy[1]/n)-x_start), 20, (255, 255, 255), -1)
    cv2.circle(res2, (50 + int(Vz[0]/n)-y_start, 50 + int(Vz[1]/n)-x_start), 20, (255, 255, 255), -1)
    cv2.line(res2, (50 + int(Vx[0] / n) - y_start, 50 + int(Vx[1] / n) - x_start),
             (50 + int(Vy[0] / n) - y_start, 50 + int(Vy[1] / n) - x_start), (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imwrite("res02.jpg", res2)



image = cv2.imread("vns.jpg")
h, w = image.shape[:2]
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edge_image = detect_edges(gray_image)

# clicked_point = []
# display_image = np.copy(edge_image)
# get_clicked_point()
# print(clicked_point)
clicked_points = [[832, 245], [1077, 211], [2062, 1245], [2816, 1170], [3070, 226], [3532, 169], [2309, 1951], [2516, 1938], [3015, 1024], [3236, 999],
                  [1379, 239], [1580, 298], [1750, 725], [1990, 785], [3545, 966], [3745, 1023], [3693, 174], [3970, 294], [3616, 1283], [3981, 1367],
                  [1655, 324], [1668, 696], [1992, 787], [2041, 2054], [2067, 1255], [2098, 2056], [2952, 471], [2971, 859], [3549, 1243], [3596, 2216]]

draw_selected_lines()
x_lines, y_lines, z_lines = group_lines_by_direction()
Vx = estimate_vanishing_point(x_lines)
Vy = estimate_vanishing_point(y_lines)
Vz = estimate_vanishing_point(z_lines)
print("Vx = \n", Vx)
print("Vy = \n", Vy)
print("Vz = \n", Vz)

homogeneous_Vx = np.array([Vx[0], Vx[1], 1])
homogeneous_Vy = np.array([Vy[0], Vy[1], 1])
horizon_line = np.cross(homogeneous_Vx, homogeneous_Vy)
draw_horizon()
draw_vanishing_points()

a, b, c = horizon_line
d = a ** 2 + b ** 2
horizon_line = horizon_line / np.sqrt(d)
a, b, c = horizon_line
d = a ** 2 + b ** 2
print("horizon = \n", horizon_line)

