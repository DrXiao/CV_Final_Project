# -*- coding: utf-8 -*-
from pathlib import Path
from typing import * 

import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.compat.v1.keras.backend import set_session

"""
env_setup()
設置tensorflow的執行環境
如果可以跑GPU就會用GPU
"""
def env_setup():
  device_name = tf.test.gpu_device_name()
  if device_name != '/device:GPU:0':
    print('GPU device not found. Will use CPU.')
  else:
    print('Found GPU at: {}'.format(device_name))

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    set_session(sess)

"""
read_h5_dataset()
從HDF5把dataset讀出來
"""
def read_h5_dataset(dataset_path):
  print("Reading datasets ...")
  dataset = h5py.File(dataset_path)
  img_shape = tuple(dataset.attrs['img-size'])
  classes = dataset.attrs['classes'][:]
  created_date = dataset.attrs['created-date']

  x_train = dataset['x_train'][:]
  y_train = dataset['y_train'][:]
  x_test = dataset['x_test'][:]
  y_test = dataset['y_test'][:]

  print("Created at ", created_date)
  print("Shape of images: ", img_shape)
  print("Classes: ", classes)
  print("Training data: ", x_train.shape, y_train.shape)
  print("Testing data: ", x_test.shape, y_test.shape)

  return (img_shape, classes), (x_train, y_train, x_test, y_test)

"""CNN"""
def create_cnn_model(img_shape, num_classes):
  model = tf.keras.applications.DenseNet121(
      input_shape = img_shape,
      include_top = False, # 是否包含全連階層
      weights = 'imagenet', # 是否載入imagenet權重
      classes = num_classes, # 類別數量
  )
  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()   # 平均池化
  dense_layer = tf.keras.layers.Dense(num_classes,activation='softmax')   # 接著一層 Softmax 的 Activation 函數
  dropout_layer = tf.keras.layers.Dropout(0.5) # 0-1之間數值防止過擬合

  cnn_model = models.Sequential()   # 順序模型是多個網絡層的線性堆疊
  cnn_model.add(model)
  cnn_model.add(global_average_layer)
  cnn_model.add(dropout_layer)
  cnn_model.add(dense_layer)
  cnn_model.summary()

  cnn_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=["categorical_accuracy"]
  )

  return cnn_model

def train_cnn_model(cnn_model, data, target, batch_size=4, epochs=20, validation_split=0.2):
  # Shuffle -> 把 data 跟 target 打亂
  index = np.arange(len(data))
  np.random.shuffle(index)
  data = data[index]
  target = target[index]

  history = cnn_model.fit(
      data, target, batch_size=batch_size, epochs=epochs,
      validation_split=validation_split
  )

  return cnn_model, history

def main():
  env_setup()

  # Read the dataset
  dataset_path = Path("./cv-final-project-cnn-dataset.h5")
  metadata, data = read_h5_dataset(dataset_path)
  img_shape, classes = metadata
  x_train, y_train, x_test, y_test = data
  num_classes = len(classes)

  cnn_model = create_cnn_model(img_shape, num_classes)
  cnn_model, history = train_cnn_model(cnn_model, x_train, y_train)
      
  # Save the trained model
  cnn_model.save(
      filepath='cnn_model.h5',
      overwrite=True,
      save_format='h5',
  )

  # Save the training history
  with open('train-history-dict.pickle', 'wb') as file_pi:
      pickle.dump(history.history, file_pi)

  # Plot the history
  _, ax=plt.subplots(2,1)
  # summarize history for accuracy
  ax[0].plot(history.history['categorical_accuracy'], label='train')
  ax[0].plot(history.history['val_categorical_accuracy'], label='val')
  ax[0].set_title('model accuracy')
  ax[0].set_ylabel('accuracy')
  ax[0].set_xlabel('epoch')
  ax[0].legend(loc='upper left')
  # summarize history for loss
  ax[1].plot(history.history['loss'], label='train')
  ax[1].plot(history.history['val_loss'], label='val')
  ax[1].set_title('model loss')
  ax[1].set_ylabel('loss')
  ax[1].set_xlabel('epoch')
  ax[1].legend(loc='upper left')

  plt.savefig('CNN.png')

  # Evaluate the model
  loss, acc = cnn_model.evaluate(x_test, y_test, batch_size=4)
  print('Tested loss=%d, accuracy='.format(loss, acc))

if __name__ == "__main__":
  main()