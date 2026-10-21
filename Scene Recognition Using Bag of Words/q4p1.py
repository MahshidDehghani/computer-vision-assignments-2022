import glob
import os
from cv2 import cv2
import numpy as np


def image_to_feature_vector(image):
    image = cv2.resize(image, (size, size))
    image = (image - np.mean(image)) / np.std(image)
    image = image.flatten()
    return image


def distance(v1, v2):
    return np.sqrt(np.sum(np.power(v1 - v2, 2)))


def knn(test_image, k=1):
    test_v = image_to_feature_vector(test_image)
    dists = []
    for c in classes:
        class_folder = train_folder + "/" + c
        for image_path in glob.glob(os.path.join(class_folder, '*.jpg')):
            img = cv2.imread(image_path, 0)
            v = image_to_feature_vector(img)
            d = distance(test_v, v)
            dists.append((d, c))
    dists = sorted(dists)
    dists = dists[:k]
    dists = np.array(dists[:k])
    dists = np.array(dists[:k])
    class_names, count = np.unique(dists[:, 1], return_counts=True)
    max_count = np.max(count)
    dict = {}
    for i in range(count.shape[0]):
        if count[i] == max_count:
            class_name = class_names[i]
            dict[class_name] = 0
    for i in range(dists.shape[0]):
        d, class_name = dists[i]
        if class_name in dict.keys():
            a = dict[class_name]
            dict[class_name] = a + d.astype(float)
    res = min(dict, key=dict.get)
    return res


def test():
    num = 0
    num_true = 0
    print()
    for c in classes:
        c_num = 0
        c_num_true = 0
        folder_path = test_folder + "/" + c
        for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
            c_num += 1
            img = cv2.imread(filename, 0)
            knn_res = knn(img)
            if knn_res == c:
                c_num_true += 1
        print(c, ": ", c_num_true / c_num)
        num_true += c_num_true
        num += c_num
    print("Accuracy: ", num_true / num)


k = 1
size = 16
train_folder = "Data/Train"
test_folder = "Data/Test"
classes = [name for name in os.listdir(train_folder)]
test()


# k =1
# Forest :  0.16
# Bedroom :  0.09
# Inside_City :  0.08
# Office :  0.08
# Highway :  0.51
# Coast :  0.56
# Industrial :  0.06
# Street :  0.54
# Livingroom :  0.18
# Suburb :  0.27
# Open_Country :  0.33
# Tall_Building :  0.29
# Mountain :  0.36
# Kitchen :  0.05
# Store :  0.02
# Accuracy:  0.23866666666666667
