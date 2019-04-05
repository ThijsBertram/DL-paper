import numpy as np
import os
from random import randint
from pathlib import Path
import time
from selenium.webdriver.common.action_chains import ActionChains
# import selenium.webdriver as webdriver
import selenium.webdriver.support.ui as ui
import pickle
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException, ElementNotInteractableException
import hashlib


with open(os.getcwd() + '\data\\' + '\data.pickle', 'rb') as pickle_in:
    data = pickle.load(pickle_in)

print(data[0])

X = []
y = []

for row in data:
    try:
        pos_val = float(row[2])
    except ValueError:
        if row[2][1] == '-':
            pos_val = -100
        else:
            pos_val = 100

    if pos_val > 0:
        y.append(1)
        X.append(row[1])
    elif pos_val < 0:
        y.append(0)
        X.append(row[1])
    else:
        continue

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)


enc = OneHotEncoder(sparse=False)
y_train_enc = np.asarray(enc.fit_transform(np.asarray(y_train).reshape((len(y_train), 1))), dtype='uint8')
y_test_enc = np.asarray(enc.fit_transform(np.asarray(y_test).reshape((len(y_test), 1))), dtype='uint8')

X_train = np.asarray([[int(x) for x in s.split(' ')] for s in X_train], dtype='int8').reshape((len(X_train), 769))
X_test = np.asarray([[int(x) for x in s.split(' ')] for s in X_test], dtype='int8').reshape((len(X_test), 769))


with open('X_train.npy', 'wb') as pickle_out:
    pickle.dump(X_train, pickle_out)

with open('X_test.npy', 'wb') as pickle_out:
    pickle.dump(X_test, pickle_out)

with open('y_train.npy', 'wb') as pickle_out:
    pickle.dump(y_train_enc, pickle_out)

with open('y_test.npy', 'wb') as pickle_out:
    pickle.dump(y_test_enc, pickle_out)
