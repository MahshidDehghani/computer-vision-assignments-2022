import glob
import os
import time
from cv2 import cv2
import numpy as np
from sklearn.cluster import KMeans


def dsift(img, step_size=10, size=16):
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(int(size/2), img.shape[0], step_size)
                                    for x in range(int(size/2), img.shape[1], step_size)]
    _,des = sift.compute(img, kp)
    return des


def create_dictionary():
    descriptors = []
    for c in classes:
        folder_path = train_data_dir + "/" + c
        for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
            img = cv2.imread(filename, 0)
            des = dsift(img)
            des_list = [row for row in des]
            descriptors.extend(des_list)
    print("kmeans")
    visual_words = KMeans(n_clusters=num_visual_words).fit(descriptors)
    file = open(visual_words_file, "wb")
    np.save(file, visual_words.cluster_centers_)
    file.close()


def corresponding_visual_word(des, visual_words):
    des_arr = np.tile(des, (num_visual_words, 1))
    dist = des_arr - visual_words
    dist = np.power(dist, 2)
    dist = np.sum(dist, axis=1)
    return np.argmin(dist)


def histogram_of_image(image, visual_words):
    des = dsift(image)
    histogram = np.zeros(num_visual_words)
    for row in des:
        index = corresponding_visual_word(row, visual_words)
        histogram[index] += 1
    normalized_histogram = [float(i) / sum(histogram) for i in histogram]
    return normalized_histogram


def save_train_histograms():
    file = open(visual_words_file, "rb")
    visual_words = np.load(file)
    file.close()
    for c in classes:
        print(c)
        folder_path = train_data_dir + "/" + c
        for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
            img = cv2.imread(filename, 0)
            histogram_name = filename.replace("Data/", "Hist2/").replace("image", "histogram").replace(".jpg", ".txt")
            histogram = histogram_of_image(img, visual_words)
            file = open(histogram_name, "wb")
            np.save(file, histogram)
            file.close()


def save_test_histograms():
    file = open(visual_words_file, "rb")
    visual_words = np.load(file)
    file.close()
    for c in classes:
        print(c)
        folder_path = test_data_dir + "/" + c
        for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
            img = cv2.imread(filename, 0)
            histogram_name = filename.replace("Data/", "Hist2/").replace("image", "histogram").replace(".jpg", ".txt")
            histogram = histogram_of_image(img, visual_words)
            file = open(histogram_name, "wb")
            np.save(file, histogram)
            file.close()


def distance(h1, h2):
    return np.sqrt(np.sum(np.power(h1 - h2, 2)))


def knn(test_histogram, k):
    dists = []
    for c in classes:
        folder_path = train_histograms_dir + "/" + c
        for histogram_name in glob.glob(os.path.join(folder_path, '*.txt')):
            file = open(histogram_name, "rb")
            histogram = np.load(file)
            file.close()
            d = distance(test_histogram, histogram)
            dists.append((d, c))
    dists = sorted(dists)
    dists = dists[:k]
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


def test(k):
    num = 0
    num_true = 0
    print()
    for c in classes:
        c_num = 0
        c_num_true = 0
        folder_path = test_histograms_dir + "/" + c
        for histogram_name in glob.glob(os.path.join(folder_path, '*.txt')):
            c_num += 1
            file = open(histogram_name, "rb")
            histogram = np.load(file)
            file.close()
            knn_res = knn(histogram, k)
            if knn_res == c:
                c_num_true += 1
        print(c, ": ", c_num_true / c_num)
        num_true += c_num_true
        num += c_num
    print("Accuracy: ", num_true / num)


sift = cv2.SIFT_create()
k = 11
num_visual_words = 50
visual_words_file = "visual_words2.txt"
train_data_dir = "Data/Train"
test_data_dir = "Data/Test"
train_histograms_dir = "Hist2/Train"
test_histograms_dir = "Hist2/Test"

classes = [name for name in os.listdir(train_data_dir)]

s = time.time()
create_dictionary()
save_train_histograms()
save_test_histograms()
test(k)
print(time.time() - s)


# k = 1
# Forest :  0.91
# Bedroom :  0.15
# Inside_City :  0.46
# Office :  0.58
# Highway :  0.6
# Coast :  0.7
# Industrial :  0.14
# Street :  0.65
# Livingroom :  0.59
# Suburb :  0.81
# Open_Country :  0.57
# Tall_Building :  0.81
# Mountain :  0.66
# Kitchen :  0.26
# Store :  0.4
# Accuracy:  0.5526666666666666