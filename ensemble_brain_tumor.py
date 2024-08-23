# -*- coding: utf-8 -*-
"""EnsembleBRT19-07

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/ensemblebrt19-07-5130a8b3-29b1-49e9-9bee-5a93202450fa.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20240823/auto/storage/goog4_request%26X-Goog-Date%3D20240823T183851Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D4fbab666ea9985e94e77f26655b186258cb8359df74e1fbe5d831353e303a4b2a8e2ba14c8965b12b1a7275464d9c7782edcca03f9935e238e2e045d30f6ab1daa96bc17ed9ceb159f51156915f90e1cef16dbc9bfb916ef15675307ff1e8dc799d9e09adf5d57876080a3152b81c575398d30dc4936d5ea23621dcf8f39a823bb35207412a21822a5be9620a08a4b88b542066cbf5a9507fbbb7368ef9ba78220033e5a0317a55a065b284777d5582719578c010dfb78ebd8fb4d841fcc4dd6b1cc558e5c5ad0968b4b55fe8b1e627180d57ccdef07ae6711f96ba0cb7b4019daa758fcaf8999763a803c24c8ba347e93600efea38cefcea9f3a0cb1bee5935
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'brain-tumor-mri-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F1608934%2F2645886%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240823%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240823T183851Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D33c5c320146204365f6e64d2d9c855258c5d527b0bf52886100f9ac2a6282ca2f8df79efd34315823b3bf0dccefbe58bfc095516f6f26f28ebcaf7dd078221834801bbdb32bb4a5abe3bf9db9b3989d67b5d5bc9b18abbab049ff5c1072d4668093a40ecca3c13593089570ab5e398ce944a83a4642df49ad9ad6db00302ca26ec4f0ac6b2237373f73202c891f47e77e80857f6c67492219c8c12f7064a98d97a098f6c47c6e673dfce6ce54be7ae72f43947961b86dae41837a32ced2f8e9423602c6f68713e7820d0ec4152f66a44a61f3e5c8eef0bee5d37681a222b332afc70d2d1b742cf6bbc65112e450c08b7dce986175878d2da49d1b1367bb3e183'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

import keras
keras.__version__
from keras import backend as K
#K.tensorflow_backend._get_available_gpus()
import tensorflow as tf

from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()

for val in local_device_protos:
  print(val.device_type)



"""## Connect to TPU

"""

import keras
keras.backend.set_session("TPU")

pip install scikit-learn

pip install seaborn

pip install plotly

pip install missingno

# Commented out IPython magic to ensure Python compatibility.

import sys
import os
import math

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['figure.dpi'] = 300
# %matplotlib inline
import seaborn as sns
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import *

from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import *

import pathlib
import tensorflow as tf

train_dir = pathlib.Path('/kaggle/input/brain-tumor-mri-dataset/Training')
test_dir = pathlib.Path('/kaggle/input/brain-tumor-mri-dataset/Testing')
img_height=224
img_width=224

train_ds  = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=128)

test_ds  = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=128)

val_ds  = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  validation_split=None,
  subset=None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=128)

"""## Visualize the data distribution"""

ROOT_DIR = r"/kaggle/input/brain-tumor-mri-dataset"
TRAIN_DIR = os.path.join(ROOT_DIR, 'Training')
TEST_DIR = os.path.join(ROOT_DIR, 'Testing')
assert os.path.isdir(ROOT_DIR) and os.path.isdir(TRAIN_DIR) and os.path.isdir(TEST_DIR)
TUMOR_CLASS = ['meningioma', 'glioma', 'pituitary', 'notumor']
IMAGE_DATA_PATHS = [os.path.join(TRAIN_DIR, tumor_class) for tumor_class in TUMOR_CLASS]
TEST_DATA_PATHS = [os.path.join(TEST_DIR, tumor_class) for tumor_class in TUMOR_CLASS]

TEST_DATA_PATHS

data_distribution_count = pd.Series([len(os.listdir(path)) for path in TEST_DATA_PATHS if os.path.exists(path) and os.path.isdir(path)],index = TUMOR_CLASS)
data_distribution_count

data_distribution_count = pd.Series([len(os.listdir(path)) for path in IMAGE_DATA_PATHS if os.path.exists(path) and os.path.isdir(path)],index = TUMOR_CLASS)
data_distribution_count

fig, axis = plt.subplots(figsize = (13, 5))
axis.grid(True, alpha = 0.1)
axis.set_title("Data Distribution Percentage (%)", fontsize = 14)
sns.barplot(x = ['\n'.join(curr_index.strip().split('_')).title() for curr_index in data_distribution_count.index],
            y = 100 * (data_distribution_count / data_distribution_count.sum()), ax = axis)
axis.set_xlabel("Tumor Class", fontsize = 12)
axis.set_ylabel("% Total Observations", fontsize = 12)
axis.tick_params(which = 'major', labelsize = 12)
axis.text(2.5, 37, f'Total Observations: {data_distribution_count.sum()}', fontdict = dict(size = 12))
sns.despine()

BRIGHTNESS_FACTOR = 1.7
fig, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (18, 5))
axes = axes.flatten()
fig.suptitle("Brain Tumor Images", fontsize = 16, fontdict = dict(weight = 'bold'), y = 1.04)
for curr_title, filename, curr_axis in zip(TUMOR_CLASS, IMAGE_DATA_PATHS, axes[:4]):
    curr_image = Image.open(os.path.join(filename, os.listdir(filename)[2]))
    img_enhancer = ImageEnhance.Brightness(curr_image)
    curr_axis.imshow(img_enhancer.enhance(BRIGHTNESS_FACTOR))
    curr_axis.set_title(" ".join(curr_title.split('_')).title(), fontsize = 14)

fig.tight_layout()
sns.despine()

"""## Image Augmentation"""

normalization_layer = tf.keras.layers.Rescaling(1./255)

import numpy as np
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

"""### 1. Inception V3"""

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Dropout
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# base_model1 = InceptionV3(
#                     input_shape=(224, 224, 3),
#                     weights='imagenet',
#                     include_top=False)
# # Freeze the first 10 layers
# for layer in base_model1.layers[:10]:
#     layer.trainable = False
# x = base_model1.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.4)(x)
# predictions1 = Dense(4, activation='softmax')(x)
# model1 = Model(inputs=base_model1.inputs, outputs=predictions1)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    base_model1 = InceptionV3(
                    input_shape=(224, 224, 3),
                    weights='imagenet',
                    include_top=False) # define your model normally
    for layer in base_model1.layers[:10]:
        layer.trainable = False
    x = base_model1.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions1 = Dense(4, activation='softmax')(x)
    model1 = Model(inputs=base_model1.inputs, outputs=predictions1)
    model1.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# model1.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

history1=model1.fit(
  train_ds,
  validation_data=test_ds,
  epochs=25)

model1.save('model1.h5')

inceptionv3_evaluation = model1.evaluate(val_ds)
inceptionv3_evaluation[1]*100

import matplotlib.pyplot as plt
acc = history1.history['accuracy']
val_acc = history1.history['val_accuracy']
loss = history1.history['loss']
val_loss = history1.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_label = np.concatenate([y for x, y in test_ds], axis=0)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Confution Matrix and Classification Report
import matplotlib.pyplot as plt
Y_pred = model1.predict_generator(test_ds, 1600)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_label, y_pred)
print(cm)
print('Classification Report')
target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(test_label, y_pred, target_names=target_names))

import seaborn as sns
sns.set(font_scale=1.0)
ax=sns.heatmap(cm, annot=True, cmap='summer', cbar=False, linewidths=3, linecolor='r', square=True, xticklabels=target_names,yticklabels=target_names,fmt='')
#sns.heatmap(cm, annot=True,annot_kws={"size": 22})
sns.set(font_scale = 2.0)
ax.set_title('Inception V3¶');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');



"""### 2. VGG16"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D

with tpu_strategy.scope():
    base_model2 = VGG16(
                        input_shape=(224, 224, 3),
                        weights='imagenet',
                        include_top=False)
    # Freeze the first 10 layers
    for layer in base_model2.layers:
        layer.trainable = False
    x = base_model2.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions2 = Dense(4, activation='softmax')(x)
    model2 = Model(inputs=base_model2.inputs, outputs=predictions2)
    model2.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# model2.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

history2=model2.fit(
  train_ds,
  validation_data=test_ds,
  epochs=25)

model2.save('model2.h5')

vgg16_evaluation = model2.evaluate(val_ds)
vgg16_evaluation[1]*100

import matplotlib.pyplot as plt
acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']
loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Confution Matrix and Classification Report
import matplotlib.pyplot as plt
Y_pred = model2.predict_generator(test_ds, 1600)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_label, y_pred)
print(cm)
print('Classification Report')
target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(test_label, y_pred, target_names=target_names))

import seaborn as sns
sns.set(font_scale=1.0)
ax=sns.heatmap(cm, annot=True, cmap='summer', cbar=False, linewidths=3, linecolor='r', square=True, xticklabels=target_names,yticklabels=target_names,fmt='')
#sns.heatmap(cm, annot=True,annot_kws={"size": 22})
sns.set(font_scale = 2.0)
ax.set_title('vgg 16');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');



"""### 3. VGG19"""

from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D

with tpu_strategy.scope():

    base_model3 = VGG19(
                        input_shape=(224, 224, 3),
                        weights='imagenet',
                        include_top=False)
    # Freeze the first 10 layers
    for layer in base_model3.layers[:10]:
        layer.trainable = False
    x = base_model3.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions3 = Dense(4, activation='softmax')(x)
    model3 = Model(inputs=base_model3.inputs, outputs=predictions3)
    model3.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# model3.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

history3=model3.fit(
  train_ds,
  validation_data=test_ds,
  epochs=25)

model3.save('model3.h5')

vgg19_evaluation = model3.evaluate(val_ds)
vgg19_evaluation[1]*100

import matplotlib.pyplot as plt
acc = history3.history['accuracy']
val_acc = history3.history['val_accuracy']
loss = history3.history['loss']
val_loss = history3.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Confution Matrix and Classification Report
import matplotlib.pyplot as plt
Y_pred = model3.predict_generator(test_ds, 1600)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_label, y_pred)
print(cm)
print('Classification Report')
target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(test_label, y_pred, target_names=target_names))

import seaborn as sns
sns.set(font_scale=1.0)
ax=sns.heatmap(cm, annot=True, cmap='summer', cbar=False, linewidths=3, linecolor='r', square=True, xticklabels=target_names,yticklabels=target_names,fmt='')
#sns.heatmap(cm, annot=True,annot_kws={"size": 22})
sns.set(font_scale = 2.0)
ax.set_title('vgg 19');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');

"""### 4. RESNET50"""

from tensorflow.keras.applications import ResNet50

with tpu_strategy.scope():

    base_model4 = ResNet50(
                        input_shape=(224, 224, 3),
                        weights='imagenet',
                        include_top=False)
    # Freeze the first 10 layers
    for layer in base_model4.layers[:10]:
        layer.trainable = False
    x = base_model4.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions4 = Dense(4, activation='softmax')(x)
    model4 = Model(inputs=base_model4.inputs, outputs=predictions4)
    model4.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# model4.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

history4=model4.fit(
  train_ds,
  validation_data=test_ds,
  epochs=25)

model4.save('model4.h5')

resnet50_evaluation = model4.evaluate(val_ds)
resnet50_evaluation[1]*100

import matplotlib.pyplot as plt
acc = history4.history['accuracy']
val_acc = history4.history['val_accuracy']
loss = history4.history['loss']
val_loss = history4.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Confution Matrix and Classification Report
import matplotlib.pyplot as plt
Y_pred = model4.predict_generator(test_ds, 1600)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_label, y_pred)
print(cm)
print('Classification Report')
target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(test_label, y_pred, target_names=target_names))

import seaborn as sns
sns.set(font_scale=1.0)
ax=sns.heatmap(cm, annot=True, cmap='summer', cbar=False, linewidths=3, linecolor='r', square=True, xticklabels=target_names,yticklabels=target_names,fmt='')
#sns.heatmap(cm, annot=True,annot_kws={"size": 22})
sns.set(font_scale = 2.0)
ax.set_title('ResNet-50');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');

"""## to write

### 5. Inception-ResNet-v2
"""

with tpu_strategy.scope():

    base_model5 = tf.keras.applications.InceptionResNetV2(
                        input_shape=(224, 224, 3),
                        weights='imagenet',
                        include_top=False)
    # Freeze the first 10 layers
    for layer in base_model5.layers[:10]:
        layer.trainable = False
    x = base_model5.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions5 = Dense(4, activation='softmax')(x)
    model5 = Model(inputs=base_model5.inputs, outputs=predictions5)
    model5.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# model5.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

history5=model5.fit(
  train_ds,
  validation_data=test_ds,
  epochs=25)

model5.save('model5.h5')

inceptionresnetv2_evaluation = model5.evaluate(val_ds)
inceptionresnetv2_evaluation[1]*100

import matplotlib.pyplot as plt
acc = history5.history['accuracy']
val_acc = history5.history['val_accuracy']
loss = history5.history['loss']
val_loss = history5.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Confution Matrix and Classification Report
import matplotlib.pyplot as plt
Y_pred = model5.predict_generator(test_ds, 1600)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_label, y_pred)
print(cm)
print('Classification Report')
target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(test_label, y_pred, target_names=target_names))

import seaborn as sns
sns.set(font_scale=1.0)
ax=sns.heatmap(cm, annot=True, cmap='summer', cbar=False, linewidths=3, linecolor='r', square=True, xticklabels=target_names,yticklabels=target_names,fmt='')
#sns.heatmap(cm, annot=True,annot_kws={"size": 22})
sns.set(font_scale = 2.0)
ax.set_title('Inception-ResNet-v2');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');

"""### 6. DenseNet-201"""

with tpu_strategy.scope():

    base_model6 = tf.keras.applications.DenseNet201(
                        input_shape=(224, 224, 3),
                        weights='imagenet',
                        include_top=False)
    # Freeze the first 10 layers
    for layer in base_model6.layers[:10]:
        layer.trainable = False
    x = base_model6.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions6 = Dense(4, activation='softmax')(x)
    model6 = Model(inputs=base_model6.inputs, outputs=predictions6)
    model6.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# model6.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

history6=model6.fit(
  train_ds,
  validation_data=test_ds,
  epochs=25)

model6.save('model6.h5')

densenet201_evaluation = model6.evaluate(val_ds)
densenet201_evaluation[1]*100

import matplotlib.pyplot as plt
acc = history6.history['accuracy']
val_acc = history6.history['val_accuracy']
loss = history6.history['loss']
val_loss = history6.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Confution Matrix and Classification Report
import matplotlib.pyplot as plt
Y_pred = model6.predict_generator(test_ds, 1600)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_label, y_pred)
print(cm)
print('Classification Report')
target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(test_label, y_pred, target_names=target_names))

import seaborn as sns
sns.set(font_scale=1.0)
ax=sns.heatmap(cm, annot=True, cmap='summer', cbar=False, linewidths=3, linecolor='r', square=True, xticklabels=target_names,yticklabels=target_names,fmt='')
#sns.heatmap(cm, annot=True,annot_kws={"size": 22})
sns.set(font_scale = 2.0)
ax.set_title('DenceNet-201');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');

"""### AlexNet

### 7. ResNet-101
"""

with tpu_strategy.scope():

    base_model7 = tf.keras.applications.ResNet101(
                        input_shape=(224, 224, 3),
                        weights='imagenet',
                        include_top=False)
    # Freeze the first 10 layers
    for layer in base_model7.layers[:10]:
        layer.trainable = False
    x = base_model7.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions7 = Dense(4, activation='softmax')(x)
    model7 = Model(inputs=base_model7.inputs, outputs=predictions7)
    model7.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# model7.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

history7=model7.fit(
  train_ds,
  validation_data=test_ds,
  epochs=25)

model7.save('model7.h5')

resnet101_evaluation = model7.evaluate(val_ds)
resnet101_evaluation[1]*100

import matplotlib.pyplot as plt
acc = history7.history['accuracy']
val_acc = history7.history['val_accuracy']
loss = history7.history['loss']
val_loss = history7.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Confution Matrix and Classification Report
import matplotlib.pyplot as plt
Y_pred = model7.predict_generator(test_ds, 1600)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_label, y_pred)
print(cm)
print('Classification Report')
target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(test_label, y_pred, target_names=target_names))

import seaborn as sns
sns.set(font_scale=1.0)
ax=sns.heatmap(cm, annot=True, cmap='summer', cbar=False, linewidths=3, linecolor='r', square=True, xticklabels=target_names,yticklabels=target_names,fmt='')
#sns.heatmap(cm, annot=True,annot_kws={"size": 22})
sns.set(font_scale = 2.0)
ax.set_title('ResNet-101');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');

"""### Mask R-CNN

### 8. MobileNet
"""

with tpu_strategy.scope():

    base_model8 = tf.keras.applications.MobileNet(
                        input_shape=(224, 224, 3),
                        weights='imagenet',
                        include_top=False)
    # Freeze the first 10 layers
    for layer in base_model8.layers[:10]:
        layer.trainable = False
    x = base_model8.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions8 = Dense(4, activation='softmax')(x)
    model8 = Model(inputs=base_model8.inputs, outputs=predictions8)
    model8.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# model8.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

history8=model8.fit(
  train_ds,
  validation_data=test_ds,
  epochs=25)

model8.save('model8.h5')

mobilenet_evaluation = model8.evaluate(val_ds)
mobilenet_evaluation[1]*100

import matplotlib.pyplot as plt
acc = history8.history['accuracy']
val_acc = history8.history['val_accuracy']
loss = history8.history['loss']
val_loss = history8.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Confution Matrix and Classification Report
import matplotlib.pyplot as plt
Y_pred = model8.predict_generator(test_ds, 1600)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_label, y_pred)
print(cm)
print('Classification Report')
target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(test_label, y_pred, target_names=target_names))

import seaborn as sns
sns.set(font_scale=1.0)
ax=sns.heatmap(cm, annot=True, cmap='summer', cbar=False, linewidths=3, linecolor='r', square=True, xticklabels=target_names,yticklabels=target_names,fmt='')
#sns.heatmap(cm, annot=True,annot_kws={"size": 22})
sns.set(font_scale = 2.0)
ax.set_title('MobileNet');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');

"""### LSTM

### GoogLeNet

### 9. Xception
"""

with tpu_strategy.scope():
    base_model9 = tf.keras.applications.Xception(
                        input_shape=(224, 224, 3),
                        weights='imagenet',
                        include_top=False)
    # Freeze the first 10 layers
    for layer in base_model9.layers[:10]:
        layer.trainable = False
    x = base_model9.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions9 = Dense(4, activation='softmax')(x)
    model9 = Model(inputs=base_model9.inputs, outputs=predictions9)
    model9.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# model9.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

history9=model9.fit(
  train_ds,
  validation_data=test_ds,
  epochs=25)

model9.save('model9.h5')

xception_evaluation = model9.evaluate(val_ds)
xception_evaluation[1]*100

import matplotlib.pyplot as plt
acc = history9.history['accuracy']
val_acc = history9.history['val_accuracy']
loss = history9.history['loss']
val_loss = history9.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Confution Matrix and Classification Report
import matplotlib.pyplot as plt
Y_pred = model9.predict_generator(test_ds, 1600)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_label, y_pred)
print(cm)
print('Classification Report')
target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(test_label, y_pred, target_names=target_names))

import seaborn as sns
sns.set(font_scale=1.0)
ax=sns.heatmap(cm, annot=True, cmap='summer', cbar=False, linewidths=3, linecolor='r', square=True, xticklabels=target_names,yticklabels=target_names,fmt='')
#sns.heatmap(cm, annot=True,annot_kws={"size": 22})
sns.set(font_scale = 2.0)
ax.set_title('Xception');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');



"""### 10. ResNet-152"""

with tpu_strategy.scope():

    base_model10 = tf.keras.applications.ResNet152(
                        input_shape=(224, 224, 3),
                        weights='imagenet',
                        include_top=False)
    # Freeze the first 10 layers
    for layer in base_model10.layers[:10]:
        layer.trainable = False
    x = base_model10.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions10 = Dense(4, activation='softmax')(x)
    model10 = Model(inputs=base_model10.inputs, outputs=predictions10)
    model10.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# model10.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

history10=model10.fit(
  train_ds,
  validation_data=test_ds,
  epochs=25)

model10.save('model10.h5')

resnet152_evaluation = model10.evaluate(val_ds)
resnet152_evaluation[1]*100

import matplotlib.pyplot as plt
acc = history10.history['accuracy']
val_acc = history10.history['val_accuracy']
loss = history10.history['loss']
val_loss = history10.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Confution Matrix and Classification Report
import matplotlib.pyplot as plt
Y_pred = model10.predict_generator(test_ds, 1600)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_label, y_pred)
print(cm)
print('Classification Report')
target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(test_label, y_pred, target_names=target_names))

import seaborn as sns
sns.set(font_scale=1.0)
ax=sns.heatmap(cm, annot=True, cmap='summer', cbar=False, linewidths=3, linecolor='r', square=True, xticklabels=target_names,yticklabels=target_names,fmt='')
#sns.heatmap(cm, annot=True,annot_kws={"size": 22})
sns.set(font_scale = 2.0)
ax.set_title('ResNet-152');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');

"""### MLP

#### Model Checkpoint
"""

from tensorflow.keras.callbacks import ModelCheckpoint
# Checkpoint to save best model per epoch

model_filepath = "/kaggle/working-{epoch:02d}-{val_accuracy:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath=model_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

"""### Building Ensemble Model"""

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
with tpu_strategy.scope():
#     model_1 = load_model('/kaggle/working/model1.h5')
#     model_1 = Model(inputs=model_1.inputs,
#                     outputs=model_1.outputs,
#                     name='name_of_model_1')

    model_2 = load_model('/kaggle/working/model2.h5')
    model_2 = Model(inputs=model_2.inputs,
                    outputs=model_2.outputs,
                    name='name_of_model_2')

    model_3 = load_model('/kaggle/working/model3.h5')
    model_3 = Model(inputs=model_3.inputs,
                    outputs=model_3.outputs,
                    name='name_of_model_3')


    model_5 = load_model('/kaggle/working/model5.h5')
    model_5 = Model(inputs=model_5.inputs,
                    outputs=model_5.outputs,
                    name='name_of_model_5')

    model_7 = load_model('/kaggle/working/model7.h5')
    model_7 = Model(inputs=model_7.inputs,
                    outputs=model_7.outputs,
                    name='name_of_model_7')

#     model_9 = load_model('/kaggle/working/model9.h5')
#     model_9 = Model(inputs=model_9.inputs,
#                     outputs=model_9.outputs,
#                     name='name_of_model_9')



    models = [model_2, model_3, model_5,model_7]
    model_input = Input(shape=(224, 224, 3))
    model_outputs = [model(model_input) for model in models]
    ensemble_output = Average()(model_outputs)
    ensemble_model = Model(inputs=model_input, outputs=ensemble_output, name='ensemble')
    ensemble_model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

# ensemble_model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

history=ensemble_model.fit(
  train_ds,


  validation_data=test_ds,
  epochs=25)

ensemble_evaluation = ensemble_model.evaluate(val_ds)
ensemble_evaluation[1]*100

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

train_label = np.concatenate([y for x, y in train_ds], axis=0)
print(train_label.shape)

test_label = np.concatenate([y for x, y in test_ds], axis=0)
print(test_label.shape)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Confution Matrix and Classification Report
import matplotlib.pyplot as plt
Y_pred = ensemble_model.predict_generator(test_ds, 1600)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_label, y_pred)
print(cm)
print('Classification Report')
target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(test_label, y_pred, target_names=target_names))

import seaborn as sns
sns.set(font_scale=1.0)
ax=sns.heatmap(cm, annot=True, cmap='summer', cbar=False, linewidths=3, linecolor='r', square=True, xticklabels=target_names,yticklabels=target_names,fmt='')
#sns.heatmap(cm, annot=True,annot_kws={"size": 22})
sns.set(font_scale = 2.0)
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');










