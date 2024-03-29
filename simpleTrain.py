import cv2
import itertools
import os
import random
import sklearn
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

images = 'images'

n_features = 646  # Change it if you add a feature
features = np.zeros((2300, n_features), np.float32)
all_shape = (640, 640, 3)
center_xy = (all_shape[0]//2 - 1, all_shape[1]//2 - 1)

trainAnswers = np.zeros((500, ), np.uint8)
with open('train') as f:
    for i, line in enumerate(f.readlines()):
        trainAnswers[i] = int(line.strip().split(',')[1] == 'True')

testAnswers = np.zeros((300, ), np.uint8)
with open('test') as f:
    for i, line in enumerate(f.readlines()):
        testAnswers[i] = int(line.strip().split(',')[1] == 'True')


numbered_images = list(enumerate(os.listdir(images)))


def hsv_histogram_example():
    '''
    improve it
    '''
    for i, name in numbered_images:
        img = cv2.imread(os.path.join(images, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img // 64
        h, s, v = cv2.split(img)
        pixels = 16 * h + 4 * s + v
        hist = np.bincount(pixels.ravel(), minlength=64)
        hist = hist.astype(float)
        # hist = np.log1p(hist)
        features[i, 64:128] = hist
##        fig = plt.figure()
##        ax = fig.add_subplot(111, projection='3d')
##        print(histo3d.shape)
##        ax.scatter(histo3d[:, 0], histo3d[:, 1], histo3d[:, 2])
##        plt.show()
##        plt.clf()
# start = 3
# bucket_size = 8
# chans = cv2.split(img)
#         color = ['b', 'g', 'r']
#         hist_needed = not os.path.isfile('Histo/{}.png'.format(name))
#         fig = plt.figure()
#         for j, ch in enumerate(chans):
#             # hist = cv2.calcHist([ch], [0], None, [bucket_size], [0, 256])
#             # features[i, start:start+bucket_size] = hist[:, 0]
#             # start += bucket_size
##        hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None,
##                            [bucket_size, bucket_size], [0, 256, 0, 256])
##        hist = np.log1p(hist)
##        features[i, start:start+bucket_size**2] = hist.flatten()
##        start += bucket_size ** 2
##        hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None,
##                            [bucket_size, bucket_size], [0, 256, 0, 256])
##        hist = np.log1p(hist)
##        features[i, start:start+bucket_size**2] = hist.flatten()
##        start += bucket_size ** 2
##        hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None,
##                            [bucket_size, bucket_size], [0, 256, 0, 256])
##        hist = np.log1p(hist)
##        features[i, start:start+bucket_size**2] = hist.flatten()
##        start += bucket_size ** 2
#             if hist_needed:
#                 ax = fig.add_subplot(211)
#                 ax.set_title('{}, answer "{}"'.format(name,
#                                                       trainAnswers[i] == 1))
#                 ax.set_xlabel('Beans')
#                 ax.set_ylabel('# of pixels')
#                 ax.plot(hist, color=color[j])
#                 ax.set_xlim([0, bucket_size-1])
#         if hist_needed:
#             # plot a 2D color histogram for green and blue
#             ax1 = fig.add_subplot(234)
#             p = ax1.imshow(hist, interpolation='nearest')
#             ax1.set_title('Green and Blue')
#
#             # plot a 2D color histogram for green and red
#             ax2 = fig.add_subplot(235)
#             p = ax2.imshow(hist, interpolation='nearest')
#             ax2.set_title('Green and Red')
#
#             # plot a 2D color histogram for blue and red
#             ax3 = fig.add_subplot(236)
#             p = ax3.imshow(hist, interpolation='nearest')
#             ax3.set_title('Blue and Red')
#         if hist_needed:
#             fig.savefig('Histo/{}.png'.format(name))
#         plt.clf()
##    for color in cv2.split(orange):
##        histo = cv2.calcHist([color], [0], None,
##                             [32], [0, 256])

    # image = cv2.imread('pokemon_games.png')
    # define the list of boundaries
##    boundaries = [
##        # red
##        ([17, 15, 100], [50, 56, 200]),
##        # blue
##        ([86, 31, 4], [220, 88, 50]),
##        # yellow
##        ([25, 146, 190], [62, 174, 250]),
##        # gray
##        ([103, 86, 65], [145, 133, 128])
##    ]
##
##    # loop over the boundaries
##    for (lower, upper) in boundaries:
##        # create NumPy arrays from the boundaries
##        lower = np.array(lower, dtype=np.uint8)
##        upper = np.array(upper, dtype=np.uint8)
##
##        # find the colors within the specified boundaries and apply the mask
##        mask = cv2.inRange(image, lower, upper)
##        output = cv2.bitwise_and(image, image, mask=mask)
##
##        # show the images
##        # cv2.imshow('images', np.hstack([image, output]))
##        # cv2.waitKey(0)

def center_pixel():
    '''
    3 features: colors of central pixel in bgr
    '''
    for i, name in numbered_images:
        img = cv2.imread(os.path.join(images, name))
        features[i, :3] = img[center_xy[0]][center_xy[1]]


def rgb_histogram_example():
    '''
    improve it
    '''
    for i, name in numbered_images:
        img = cv2.imread(os.path.join(images, name))
        img = img // 64
        b, g, r = cv2.split(img)
        pixels = 16 * r + 4 * g + b  # r << 4 + g << 2 + b
        hist = np.bincount(pixels.ravel(), minlength=64)
        hist = hist.astype(float)
        hist = np.log1p(hist)
        features[i, 3:67] = hist


def color_detect():
    '''
    TODO: add cv2.inRange method call with HSV color space
    '''
    def salt_and_pepper(channel):
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                rand = random.random()
                if rand < 0.15:
                    channel[i][j] = 0
                elif rand > 0.85:
                    channel[i][j] = 255
        return channel

    def chi2(P, Q, eps=1e-10):
        return 0.5 * np.sum(((p - q) ** 2) / (p + q + eps)
                            for (p, q) in zip(P, Q))

    def get_orange_salted():
        orange = np.zeros(all_shape, np.uint8)
        orange[:, :, 1] += 165
        orange[:, :, 2] += 255
        return cv2.merge(map(salt_and_pepper, cv2.split(orange)))
    orange = cv2.medianBlur(get_orange_salted(), 3)
    orange_histo = cv2.calcHist([orange], [0, 1, 2], None,
                               [8, 8, 8], [0, 256, 0, 256, 0, 256])

    for i, name in numbered_images:
        img = cv2.imread(os.path.join(images, name))
        img_histo = cv2.calcHist([img], [0, 1, 2], None,
                                 [8, 8, 8], [0, 256, 0, 256, 0, 256])
        features[i][67] = chi2(img_histo.flatten(), orange_histo.flatten())
        orange_hsv = (np.array([15, 100, 100], dtype=np.uint8),
                      np.array([55, 255, 255], dtype=np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img, orange_hsv[0], orange_hsv[1])
        filtered = cv2.bitwise_and(img, img, mask=mask)
        # if not (i % 100):
        # cv2.imshow(str(i), filtered)
        # cv2.waitKey(0)
        _, countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        average_area = np.average([a for a in map(cv2.contourArea, countours)
                                   if a > 12])
        features[i][68] = average_area if not np.isnan(average_area) else 0.0
        average_circle = np.average([radius for (x, y), radius in map(cv2.minEnclosingCircle, countours)
                                   if 5 < radius < 420])
        features[i][69] = average_circle if not np.isnan(average_circle) else 0.0


def histogram_simple():
    '''
    right now added only mean of the histogram :(
    3D histogram, three 2D histograms
    '''
    for i, name in numbered_images:
        img = cv2.imread(os.path.join(images, name))
        histo3d = cv2.calcHist([img], [0, 1, 2], None,
                               [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histo3d = np.log1p(histo3d)
        features[i, 70:582] = histo3d.flatten()


def twod_histo():
    for i, name in numbered_images:
        img = cv2.imread(os.path.join(images, name))
        chans = cv2.split(img)
        hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None,
                            [8, 8], [0, 256, 0, 256])
        hist = np.log1p(hist)
        features[i, 582:] = hist.flatten()


# here is great place for new features
# please do it in function and
# write a brief comment with a description at the top

# at next function we must write features[i, 3:]
# because we already has first 3 features

center_pixel()
histogram_simple()
rgb_histogram_example()
# hsv_histogram_example()
twod_histo()
color_detect()
# do not forget to exec your code

# cv2.normalize(features, features)

# training time!
trainFeatures = features[0:500]
testFeatures = features[500:800]
evalFeatures = features[800:2300]

X = trainFeatures
y = trainAnswers

##names = ['Nearest Neighbors', # 'Linear SVM', 'RBF SVM',
##         'Decision Tree',
##         'Random Forest', 'AdaBoost', 'Naive Bayes',
##         'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis']
##
##classifiers = [
##    KNeighborsClassifier(),
##    # SVC(kernel='linear'),
##    # SVC(),
##    DecisionTreeClassifier(),
##    RandomForestClassifier(n_estimators=250),
##    AdaBoostClassifier(),
##    GaussianNB(),
##    LinearDiscriminantAnalysis(),
##    QuadraticDiscriminantAnalysis()
##]
##
##for name, clf in zip(names, classifiers):
##        clf.fit(X, y)
##        tested = clf.predict(testFeatures)
##        print(name, sklearn.metrics.f1_score(testAnswers, tested))

clf = GaussianNB()
clf.fit(X, y)
tested = clf.predict(testFeatures)
print(sklearn.metrics.f1_score(testAnswers, tested))
#
evaluated = clf.predict(evalFeatures)
with open('results_NG_AL_NN.csv', 'w') as f:
    f.write('Id,Prediction\n')
    for i in range(800, 2300):
        filename = 'images/{:04}.jpg'.format(i)
        f.write(','.join((filename, str(evaluated[i - 800]))) + '\n')

