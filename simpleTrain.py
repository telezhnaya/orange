import cv2
import os
import sklearn
import sklearn.ensemble
import numpy as np


images = 'images'

n_features = 3  # Change it if you add a feature
features = np.zeros((2300, n_features), np.uint8)

trainAnswers = np.zeros((500, ), np.uint8)
with open("train") as f:
    for i, line in enumerate(f.readlines()):
        trainAnswers[i] = int(line.strip().split(',')[1] == 'True')

testAnswers = np.zeros((300, ), np.uint8)
with open("test") as f:
    for i, line in enumerate(f.readlines()):
        testAnswers[i] = int(line.strip().split(',')[1] == 'True')


# 3 features: colors of central pixel in bgr
def features012_example():
    for i, o in enumerate(os.listdir(images)):
        imagePath = os.path.join(images, o)
        img = cv2.imread(imagePath)
        if o != '0613.jpg':
            features[i] = img[319][319]
            print(o)
        else:
            features[i] = features[-1]

# here is great place for new features
# please do it in function and
# write a brief comment with a description at the top

# at next function we must write features[i, 3:]
# because we already has first 3 features

features012_example()
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
