from read_dataset import *
from sklearn import svm

def create_svm_model(data: list, target: list, kernel: str = "rbf") -> svm.SVC:
    svm_model = svm.SVC(kernel=kernel, gamma="auto")
    svm_model.fit(data, target)

    return svm_model

def main():
    train_dataset, test_dataset = read_dataset()
    
    train_data, train_target = split_data_target(train_dataset)
    train_target = labelsname_to_labelsnum(train_target)
    test_data, test_target = split_data_target(test_dataset)
    test_target = labelsname_to_labelsnum(test_target)

    kernel = "linear"
    #for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    svm_model = create_svm_model(train_data, train_target, kernel)
    print("--- %s ---"%(kernel))
    print(svm_model)
    print(svm_model.score(train_data, train_target))
    print(svm_model.score(test_data, test_target))
    labels = svm_model.predict(test_data)
    print(labelsnum_to_labelsname(labels))


if __name__ == "__main__":
    main()