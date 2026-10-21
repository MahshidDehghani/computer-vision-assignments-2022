import glob
import os
import time

from cv2 import cv2
import numpy as np
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
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
    hist = np.zeros(num_visual_words)
    for row in des:
        index = corresponding_visual_word(row, visual_words)
        hist[index] += 1
    norm_hist = [float(i) / sum(hist) for i in hist]
    return norm_hist


def save_train_histograms():
    file = open(visual_words_file, "rb")
    visual_words = np.load(file)
    file.close()
    for c in classes:
        print(c)
        folder_path = train_data_dir + "/" + c
        for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
            img = cv2.imread(filename, 0)
            histogram_name = filename.replace("Data/", "Hist3/").replace("image", "histogram").replace(".jpg", ".txt")
            hist = histogram_of_image(img, visual_words)
            file = open(histogram_name, "wb")
            np.save(file, hist)
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
            histogram_name = filename.replace("Data/", "Hist3/").replace("image", "histogram").replace(".jpg", ".txt")
            hist = histogram_of_image(img, visual_words)
            file = open(histogram_name, "wb")
            np.save(file, hist)
            file.close()


def save_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion matrix')
    plt.colorbar()
    index_array = np.arange(len(classes))
    plt.xticks(index_array, classes, rotation=60)
    plt.yticks(index_array, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("res09.jpg")


def get_train_histograms():
    X = []
    labels = []
    class_idx = 0
    for c in classes:
        folder_path = train_histograms_dir + "/" + c
        for histogram_name in glob.glob(os.path.join(folder_path, '*.txt')):
            file = open(histogram_name, "rb")
            hist = np.load(file)
            file.close()
            X.append(hist)
            labels.append(class_idx)
        class_idx += 1
    return np.asarray(X), labels


def test():
    confusion_matrix = np.zeros((15, 15))
    total_count = 0
    total_correct = 0
    print()
    class_idx = 0
    for c in classes:
        class_count = 0
        correct_count = 0
        folder_path = test_histograms_dir + "/" + c
        for histogram_name in glob.glob(os.path.join(folder_path, '*.txt')):
            class_count += 1
            file = open(histogram_name, "rb")
            hist = np.load(file)
            file.close()
            svm_res = classifier.predict(np.asarray([hist]))
            confusion_matrix[class_idx, svm_res] = confusion_matrix[class_idx, svm_res] + 1
            if svm_res == class_idx:
                correct_count += 1
        confusion_matrix[class_idx, :] = confusion_matrix[class_idx, :] / class_count
        print(c, ": ", correct_count / class_count)
        total_correct += correct_count
        total_count += class_count
        class_idx += 1
    print("Accuracy: ", total_correct / total_count)
    # print(confusion_matrix)
    save_confusion_matrix(confusion_matrix)


sift = cv2.SIFT_create()
visual_words_file = "visual_words3.txt"
num_visual_words = 400
train_data_dir = "Data/Train"
test_data_dir = "Data/Test"
train_histograms_dir = "Hist3/Train"
test_histograms_dir = "Hist3/Test"
classes = [name for name in os.listdir(train_data_dir)]


train_histograms, train_labels = get_train_histograms()
classifier = LinearSVC(C=250)
classifier.fit(train_histograms, train_labels)

s = time.time()
# create_dictionary()
# save_train_histograms()
# save_test_histograms()
test()
print(time.time() - s)

# c = 10
# Forest :  0.95
# Bedroom :  0.43
# Inside_City :  0.7
# Office :  0.7
# Highway :  0.79
# Coast :  0.83
# Industrial :  0.48
# Street :  0.72
# Livingroom :  0.7
# Suburb :  0.89
# Open_Country :  0.75
# Tall_Building :  0.86
# Mountain :  0.86
# Kitchen :  0.41
# Store :  0.7
# Accuracy:  0.718