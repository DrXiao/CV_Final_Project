import numpy as np
from pathlib import Path
from typing import * 
import os

file_path = Path("final report - 1")

name_file = Path("name.txt")
fnormal_file = Path("fnormal.txt")
query_file = Path("query.txt")

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

def main():
    train_dataset, test_dataset = read_dataset()
    
    print(classes)
    print(labelnum_to_labelname(2))
    print(labelname_to_labelnum("Beach"))


if __name__ == "__main__":
    main()
