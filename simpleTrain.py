import cv2
import os
import random
import sklearn
import sklearn.ensemble
import numpy as np
from matplotlib import pyplot as plt

images = 'images'

n_features = 64  # Change it if you add a feature
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


def center_pixel():
    '''
    3 features: colors of central pixel in bgr
    '''
    for i, name in numbered_images:
        img = cv2.imread(os.path.join(images, name))
        if name != '0613.jpg':
            features[i, :3] = img[center_xy[0]][center_xy[1]]
            if not i % 100:
                print(name)
        else:
            features[i] = features[-1]


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

    def get_orange_salted():
        orange = np.zeros(all_shape, np.uint8)
        orange[:, :, 1] += 165
        orange[:, :, 2] += 255
        return cv2.merge(map(salt_and_pepper, cv2.split(orange)))
    orange = cv2.medianBlur(get_orange_salted(), 3)

    for color in cv2.split(orange):
        histo = cv2.calcHist([color], [0], None,
                             [32], [0, 256])

    image = cv2.imread('pokemon_games.png')
    # define the list of boundaries
    boundaries = [
        # red
        ([17, 15, 100], [50, 56, 200]),
        # blue
        ([86, 31, 4], [220, 88, 50]),
        # yellow
        ([25, 146, 190], [62, 174, 250]),
        # gray
        ([103, 86, 65], [145, 133, 128])
    ]

    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # show the images
        # cv2.imshow("images", np.hstack([image, output]))
        # cv2.waitKey(0)


def histogram_simple():
    '''
    right now added only mean of the histogram :(
    3D histogram, three 2D histograms
    '''
    plt.figure()
    plt.title('test')
    plt.xlabel('Beans')
    plt.ylabel('# of pixels')
    '''
                plt.plot(histo)
                plt.show()
                cv2.waitKey(0)
    '''
    for i, name in numbered_images:
        img = cv2.imread(os.path.join(images, name))
        if name != '0613.jpg':
            histo = cv2.calcHist([img], [0, 1, 2], None,
                                 [8, 8, 8], [0, 256, 0, 256, 0, 256])
            features[i][3] = np.mean(histo.flatten())
            if not i % 100:
                print(name)
        else:
            features[i] = features[-1]


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
        features[i] = hist


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
        hist = np.log1p(hist)
        features[i] = hist


# here is great place for new features
# please do it in function and
# write a brief comment with a description at the top

# at next function we must write features[i, 3:]
# because we already has first 3 features

# center_pixel()
# color_detect()
# histogram_simple()
rgb_histogram_example()
# hsv_histogram_example()
# do not forget to exec your code


# training time!
trainFeatures = features[0:500]
testFeatures = features[500:800]
evalFeatures = features[800:2300]

X = trainFeatures
y = trainAnswers

clf = sklearn.ensemble.RandomForestClassifier()
clf.fit(X, y)

tested = clf.predict(testFeatures)
print(sklearn.metrics.f1_score(testAnswers, tested))
# gives 7.391

evaluated = clf.predict(evalFeatures)
with open('results.csv', 'w') as f:
    f.write('Id,Prediction\n')
    for i in range(800, 2300):
        filename = 'images/{:04}.jpg'.format(i)
        f.write(','.join((filename, str(evaluated[i - 800]))) + '\n')
