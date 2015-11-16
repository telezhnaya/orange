import cv2
import os
import sklearn

images = 'images'

# creating features, simple one = color of central pixel

features = []

for o in os.listdir(images):
    imagePath = os.path.join(images, o)
    img = cv2.imread(imagePath)
    print o
    if o != '0613.jpg':
        features.append(img[319][319])
    else:
        features.append(features[-1])

trainFeatures = features[0:500]
testFeatures = features[500:800]
evalFeatures = features[800:2300]

trainAnswers = []
f = open("train")
train = f.readlines()
f.close()

for line in train:
    trainAnswers.append(int(line.strip().split(',')[1] == 'True'))

testAnswers = []
f = open("test")
test = f.readlines()
f.close()

for line in test:
    testAnswers.append(int(line.strip().split(',')[1] == 'True'))

# training time!

X = trainFeatures
y = trainAnswers

clf = sklearn.ensemble.RandomForestClassifier()
clf.fit(X, y)

tested = clf.predict(testFeatures)
print(sklearn.metrics.f1_score(testAnswers, tested))
# gives 7.391


evaluated = clf.predict(evalFeatures)
f = open('baseline.csv', 'w')
f.write("Id,Prediction\n")
for i in range(800, 2300):
    filename = "images/" + "0" * (4 - len(str(i))) + str(i) + ".jpg"
    f.write(filename + "," + str(evaluated[i - 800]) + "\n")
f.close()
