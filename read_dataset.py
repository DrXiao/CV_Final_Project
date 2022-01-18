from cgi import test
import numpy as np
from pathlib import Path
from typing import * 
import os
import cv2
import matplotlib.pyplot as plt

file_path = Path("final report - 1")

name_file = Path("name.txt")
fnormal_file = Path("fnormal.txt")
query_file = Path("query.txt")
picture_dir = file_path / "pic"
img_shape = (224, 224, 3)

classes = sorted([
    dir_name for dir_name in os.listdir(file_path / "pic")
])

def read_dataset(name_file: str = file_path / name_file, fnormal_file: str = file_path / fnormal_file, query_file: str = file_path/query_file) -> Tuple[list, list]:
    all_imgs_features = []
    with open(name_file, "r") as name, open(fnormal_file, "r") as fnorm, \
            open(query_file, "r") as query:
        all_name = name.readlines()
        all_fnorm = fnorm.readlines()
        all_query = [int(i) for i in query.readlines()]
        for i in range(len(all_name)):
            name, img_class = all_name[i].split(" ")[:2]
            img_class = img_class.split("\n")[0]
            fnorm = np.array([float(num) for num in all_fnorm[i].split()])
            all_imgs_features.append([name, img_class, fnorm])

    train_dataset = []
    test_dataset = [all_imgs_features[idx - 1] for idx in all_query]

    for i in range(len(all_imgs_features)):
        if i + 1 not in all_query:
            train_dataset.append(all_imgs_features[i])

    return train_dataset, test_dataset

def split_data_target(dataset: list) -> Tuple[list, list]:
    data = []
    target = []

    for img in dataset:
        data.append(img[2])
        target.append(img[1])
    
    return data, target

def labelnum_to_labelname(label_num: int) -> str:
    return classes[label_num]

def labelsnum_to_labelsname(labels_num: List[int]) -> List[str]:
    return [classes[label_num] for label_num in labels_num]

def labelname_to_labelnum(label_name: str) -> int:
    return classes.index(label_name)

def labelsname_to_labelsnum(labels_name: List[str]) -> List[int]:
    return [classes.index(label_name) for label_name in labels_name]

def read_imgs(name_file: str = file_path / name_file, fnormal_file: str = file_path / fnormal_file, query_file: str = file_path/query_file) -> Tuple[list, list]:
    all_imgs_features = []
    print("Reading images...")
    with open(name_file, "r") as name, open(fnormal_file, "r") as fnorm, \
            open(query_file, "r") as query:
        all_name = name.readlines()
        all_fnorm = fnorm.readlines()
        all_query = [int(i) for i in query.readlines()]
        for i in range(len(all_name)):
            name, img_class = all_name[i].split(" ")[:2]
            img_class = img_class.split("\n")[0]
            img_path = str((picture_dir/img_class/name).resolve())
            try:
                # cv2.imread() 讀某些照片的時候會有問題 (e.g. Planet/Planet_027.jpg)
                # Ref: https://stackoverflow.com/questions/54265107/cv2-imread-returns-none
                img = plt.imread(img_path)
                img = cv2.resize(img, img_shape[:2], interpolation=cv2.INTER_NEAREST)

                # 因為照片有2, 3, 4通道，所以要處理成統一都是3通道
                if(len(img.shape) == 2): # For mono color image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif(img.shape[2] == 4): # For images with alpha channel
                    img = img[:,:,:3]
                assert img.shape == img_shape
            except Exception as e:
                print(img_path, e)
                break
            all_imgs_features.append([name, img_class, img])

    train_dataset = []
    test_dataset = [all_imgs_features[idx - 1] for idx in all_query]

    for i in range(len(all_imgs_features)):
        if i + 1 not in all_query:
            train_dataset.append(all_imgs_features[i])

    return train_dataset, test_dataset

def main():
    train_dataset, test_dataset = read_dataset()
    
    print(classes)
    print(labelnum_to_labelname(2))
    print(labelname_to_labelnum("Beach"))

    train_dataset, test_dataset = read_imgs()
    print(test_dataset[0][:2])
    plt.imshow(test_dataset[0][2])
    plt.show()

if __name__ == "__main__":
    main()
