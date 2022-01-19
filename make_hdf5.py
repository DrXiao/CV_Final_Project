import h5py
import read_dataset
from tensorflow.keras.utils import to_categorical
import numpy as np
from pathlib import Path
from datetime import datetime

out_file_name = Path('cv-final-project-cnn-dataset.h5')

def main():
    train_dataset, test_dataset = read_dataset.read_imgs()

    # features & label
    x_train, y_train = read_dataset.split_data_target(train_dataset) 
    y_train = read_dataset.labelsname_to_labelsnum(y_train)
    x_test, y_test = read_dataset.split_data_target(test_dataset)
    y_test = read_dataset.labelsname_to_labelsnum(y_test)

    # One-hot encoding
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    x_train = np.array(x_train)
    y_train_hot = np.array(y_train_hot)
    x_test = np.array(x_test)
    y_test_hot = np.array(y_test_hot)

    with h5py.File(out_file_name, "w") as out:
        # metadata 和資料集無關的資訊 
        out.attrs['classes'] = read_dataset.classes 
        out.attrs['img-size'] = read_dataset.img_shape
        out.attrs['created-date'] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        # x: 圖片； y: label
        out['x_train'] = x_train
        out['y_train'] = y_train_hot
        out['x_test'] = x_test
        out['y_test'] = y_test_hot
    
    print("Training data: ", x_train.shape, y_train_hot.shape)
    print("Testing data: ", x_test.shape, y_test_hot.shape)
    print("Classes: ", read_dataset.classes)
    print("Generated HDF5 file: ", out_file_name, " Size=", out_file_name.stat().st_size//1024//1024, "MB")
    # 用來看產生的檔案多大

if __name__ == "__main__":
    main()