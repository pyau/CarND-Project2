# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file='./traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np
import pandas as pd

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# TODO: How many unique classes/labels there are in the dataset.
df = pd.read_csv('./signnames.csv')
n_classes = df.shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
# Visualizations will be shown in the notebook.
#%matplotlib inline

num_rows = 4
num_cols = 5
num_plots = num_rows*num_cols

plt.figure(1, figsize=(8,8))
for i in range(0, num_plots):
    im=X_train[random.randint(1, n_train)]
    plt.subplot(num_rows,num_cols, i+1)
    plt.imshow(im)
    plt.axis('off')
plt.savefig("signs_ex.png", bbox_inches='tight')


### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
import cv2

X_train_proc = np.zeros([X_train.shape[0], X_train.shape[1], X_train.shape[2]])
#X_res = np.zeros([X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]])
i = 0
X_train, y_train = shuffle(X_train, y_train)
for img in X_train[:]:
    blur = cv2.GaussianBlur(img,(5,5),0)
    res = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
#    res = blur
    X_train_proc[i] = res / 255.0 - 0.5
    i += 1


plt.figure(1, figsize=(8,8))
for i in range(0, num_plots):
    im=X_train_proc[random.randint(1, n_train)]
    plt.subplot(num_rows,num_cols, i+1)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
plt.savefig("signs_grey.png", bbox_inches='tight')

