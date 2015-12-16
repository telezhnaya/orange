import os
import cv2
import pandas as pd
import numpy as np
import sklearn
from functools import reduce, wraps
from itertools import islice
from scipy.cluster.vq import vq, kmeans
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def iter_images():
    for i in range(2300):
        img_path = "images/{:04}.jpg".format(i)
        img = cv2.imread(img_path)
        yield img


def read_answers(path):
    csv = pd.read_csv(path, header=None)
    answers = csv.values[:, 1].astype(int)
    return answers


def write_answers(path, answers):
    id = ["images/{:04}.jpg".format(i + 800) for i in range(len(answers))]
    df = pd.DataFrame({"Id": id, "Prediction": answers})
    df.to_csv(path, index=False, header=True)


def dumped(func):
    @wraps(func)
    def inner(*args, **kwargs):
        path = "dump/{}.pkl".format(func.__name__[5:])
        if os.path.exists(path):
            result = joblib.load(path)
        else:
            result = func(*args, **kwargs)
            joblib.dump(result, path, compress=3)
        return result
    return inner


@dumped
def calc_rgb_hists():
    features = []
    for img in iter_images():
        hist = cv2.calcHist([img], [0, 1, 2], None, [4, 4, 4],
                            [0, 256, 0, 256, 0, 256])
        features.append(hist.ravel())
    return features


@dumped
def calc_hsv_hists():
    features = []
    for img in iter_images():
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([img], [0, 1, 2], None, [4, 4, 4],
                            [0, 256, 0, 256, 0, 256])
        features.append(hist.ravel())
    return features


@dumped
def calc_surf_bow():
    surf = cv2.xfeatures2d.SURF_create(500)
    descriptor = []
    for img in islice(iter_images(), 400):
        img = cv2.medianBlur(img, 15)
        # img = cv2.resize(img, (256, 256))
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
        kp, des = surf.compute(img, surf.detect(edges))
        descriptor.extend(des)

    k = 200
    codebook, distortion = kmeans(descriptor, k, 1)
    del descriptor

    features = np.zeros((2300, k), np.float32)
    for i, img in enumerate(iter_images()):
        img = cv2.medianBlur(img, 15)
        # img = cv2.resize(img, (256, 256))
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
        kp = surf.detect(edges)
        kp, des = surf.compute(img, kp)
        code, dist = vq(des, codebook)
        features[i][code] += 1

    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)

    return features


@dumped
def calc_items():
    items = []
    for img in iter_images():
        img = cv2.resize(img, (64, 64))
        items.append(img.ravel())
    return items


def train_classifiers(X, y, train_thresh=500):
    classifiers = {
        "Random Forest 25": RandomForestClassifier(25),
        "Random Forest 50": RandomForestClassifier(50),
        "Random Forest 100": RandomForestClassifier(100),
        "Random Forest 500": RandomForestClassifier(500),
        "Random Forest 1000": RandomForestClassifier(1000),
        "AdaBoost": AdaBoostClassifier()
    }

    train_features = X[:train_thresh]
    test_features = X[train_thresh:]
    train_answers = y[:train_thresh]
    test_answers = y[train_thresh:]

    max_score = 0
    for clf_name, clf in classifiers.items():
        clf.fit(train_features, train_answers)
        tested = clf.predict(test_features)
        score = sklearn.metrics.accuracy_score(test_answers, tested)
        print(clf_name, score)
        if max_score < score:
            max_score = score
            best_clf = clf

    return best_clf


def main():
    features = []

    rgb_hists = calc_rgb_hists()
    hsv_hists = calc_hsv_hists()
    # surf_bow = calc_surf_bow()
    # items = calc_items()

    features.append(rgb_hists)
    features.append(hsv_hists)
    # features.append(surf_bow)
    # features.append(items)

    features = reduce(lambda a, b: np.hstack((a, b)), features)
    train_features = features[:800]
    eval_features = features[800:]

    train_answers = read_answers("train")
    test_answers = read_answers("test")
    train_answers = np.concatenate((train_answers, test_answers))

    clf = train_classifiers(train_features, train_answers, 600)
    eval_answers = clf.predict(eval_features)
    write_answers("results.csv", eval_answers)


if __name__ == "__main__":
    main()
