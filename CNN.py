from read_dataset import * 
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle

def create_cnn_model(data, target):
    batch_size = 4
    epochs = 1
    num_classes = 50
    validation_split = 0.2
    model = tf.keras.applications.DenseNet121(
        input_shape = (224, 224, 3),
        include_top = False, # 是否包含全連階層
        weights='imagenet', # 是否載入imagenet權重
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

    cnn_model.compile(loss='categorical_crossentropy'
                  ,optimizer='adam',metrics=["accuracy"])

    print(data.shape, target.shape)

    history = cnn_model.fit(
        data, target, batch_size=batch_size, epochs=epochs,
        validation_split=validation_split
    )
    return cnn_model, history

def main():
    train_dataset, test_dataset = read_imgs()

    # features & label
    x_train, y_train = split_data_target(train_dataset) 
    y_train = labelsname_to_labelsnum(y_train)
    x_test, y_test = split_data_target(test_dataset)
    y_test = labelsname_to_labelsnum(y_test)

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    #x_train = np.array(x_train)
    #y_train = np.array(y_train)

    # Shuffle the training dataset
    index = np.arange(len(x_train))
    np.random.shuffle(index)
    x_train = np.array(x_train)[index]
    y_train_hot = np.array(y_train_hot)[index]

    print(x_train, y_train_hot)

    cnn_model, history = create_cnn_model(x_train, y_train_hot)
    
    cnn_model.save(
        filepath='model/',
        overwrite=True,
        save_format='tf',
    )

    #cnn_model.summary()

    with open('trainHistoryDict.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    fig,ax=plt.subplots(2,1)
    # summarize history for accuracy
    ax[0].plot(history.history['accuracy'], label='train')
    ax[0].plot(history.history['val_accuracy'], label='val')
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

    score, acc = cnn_model.evaluate(x_test, y_test_hot, batch_size=4)

    print('Test', score, acc)

if __name__=="__main__":
    main()