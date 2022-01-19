from read_dataset import *
import numpy as np
import keras
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

def create_cnn_model(data: list, target: list) -> Sequential:
    cnn_model = Sequential()

    new_shape = (6, 37, 1)
    cnn_model.add(Conv2D(32, 3, 3, input_shape = new_shape, activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation = 'relu'))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Dense(50, activation = 'sigmoid'))
    cnn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    cnn_model.summary()
    history = cnn_model.fit(data, target, verbose=2, epochs=200, batch_size=200)
    loss = history.history['loss']
    acc = history.history['accuracy']

    '''圖形印出loss and acc '''
    plt.title('loss and acc')
    plt.plot(loss)
    plt.plot(acc)
    plt.legend(['Loss', 'Acc'])
    plt.xlabel('epochs')
    plt.show()
    return cnn_model

def main():
    train_dataset, test_dataset = read_dataset()
    
    train_data, train_target = split_data_target(train_dataset)
    train_target = labelsname_to_labelsnum(train_target)
    test_data, test_target = split_data_target(test_dataset)
    test_target = labelsname_to_labelsnum(test_target)
    train_data = np.array(train_data).reshape(7000, 6, 37, 1)
    test_data = np.array(test_data).reshape(3000, 6, 37, 1)
    train_hot = to_categorical(train_target)
    test_hot = to_categorical(test_target)

    # kernel = "linear"
    cnn_model = create_cnn_model(train_data, train_hot)


    print(cnn_model)

    labels = cnn_model.predict(train_data)
    
    cnt = 0
    print("== train ==")
    for idx in range(len(labels)):
        label = np.argmax(labels[idx])
        if label == train_target[idx]:
            cnt += 1
    print(cnt)
    print(cnt / len(labels) * 100, "%")

    labels = cnn_model.predict(test_data)
    cnt = 0
    print("== test ==")
    for idx in range(len(labels)):
        label = np.argmax(labels[idx])
        if label == test_target[idx]:
            cnt += 1
    print(cnt)
    print(cnt / len(labels) * 100, "%")
    

if __name__ == "__main__":
    main()
