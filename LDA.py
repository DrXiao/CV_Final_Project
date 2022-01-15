import numpy as np
import read_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Data Type
# train_dataset[][0]: picture name
# train_dataset[][1]: picture label
# train_dataset[][2]: picture feature
###

train_dataset, test_dataset = read_dataset.read_dataset()
tr = []
cls = []
for i in range(len(train_dataset)):
    tr.append(train_dataset[i][2])
    cls.append(train_dataset[i][1])
clf = LinearDiscriminantAnalysis()
clf.fit(tr, cls)


cnt = 0
for i in range(len(test_dataset)):
    print("Acturally class = {}, predict class = {}".format(
        test_dataset[i][1], clf.predict([test_dataset[i][2]])))
    if clf.predict([test_dataset[i][2]]) == test_dataset[i][1]:
        cnt += 1

print(cnt/len(test_dataset)*100, "%")
