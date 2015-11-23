import cv2
import os
import sklearn
import sklearn.ensemble
import numpy as np


images = 'images'

n_features = 4  # Change it if you add a feature
features = np.zeros((2300, n_features), np.uint8)
all_shape = (640, 640, 3)

trainAnswers = np.zeros((500, ), np.uint8)
with open("train") as f:
    for i, line in enumerate(f.readlines()):
        trainAnswers[i] = int(line.strip().split(',')[1] == 'True')

testAnswers = np.zeros((300, ), np.uint8)
with open("test") as f:
    for i, line in enumerate(f.readlines()):
        testAnswers[i] = int(line.strip().split(',')[1] == 'True')


numbered_images = enumerate(os.listdir(images))


# 3 features: colors of central pixel in bgr
def center_pixel():
    for i, name in numbered_images:
        img = cv2.imread(os.path.join(images, name))
        if name != '0613.jpg':
            features[i, :3] = img[all_shape[0]//2 - 1][all_shape[1]//2 - 1]
            if not i % 100:
                print(name)
        else:
            features[i] = features[-1]


def color_detect():
    image = cv2.imread('pokemon_games.png')

    def get_orange_salted():
        orange = np.zeros(all_shape, np.uint8)
        orange += (255, 165, 0)
        # print(orange)
        # cv2.imshow("images", orange)
    get_orange_salted()
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

# here is great place for new features
# please do it in function and
# write a brief comment with a description at the top

# at next function we must write features[i, 3:]
# because we already has first 3 features

# center_pixel()
color_detect()
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
with open('baseline.csv', 'w') as f:
    f.write("Id,Prediction\n")
    for i in range(800, 2300):
        filename = "images/" + "0" * (4 - len(str(i))) + str(i) + ".jpg"
        f.write(filename + "," + str(evaluated[i - 800]) + "\n")
