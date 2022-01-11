from read_dataset import *
from sklearn.ensemble import RandomForestClassifier

def create_rand_forest(data: list, target: list) -> RandomForestClassifier:
    rand_forest_model = RandomForestClassifier()
    rand_forest_model.fit(data, target)
    
    return rand_forest_model

def main():
    train_dataset, test_dataset = read_dataset()
    
    train_data, train_target = split_data_target(train_dataset)
    train_target = labelsname_to_labelsnum(train_target)
    test_data, test_target = split_data_target(test_dataset)
    test_target = labelsname_to_labelsnum(test_target)

    kernel = "linear"
    rand_forest_model = create_rand_forest(train_data, train_target)
    print("--- %s ---"%(kernel))
    print(rand_forest_model)
    print(rand_forest_model.score(train_data, train_target))
    print(rand_forest_model.score(test_data, test_target))
    labels = rand_forest_model.predict(test_data)
    #print(labelsnum_to_labelsname(labels))

    print(len(rand_forest_model.feature_importances_))


if __name__ == "__main__":
    main()