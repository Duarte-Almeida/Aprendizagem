import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import feature_selection, model_selection, tree
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

def load_data(filename):

    dataset = arff.loadarff(filename)
    dataset = pd.DataFrame(dataset[0])

    # remove b's in "class" and instances with missing features
    dataset[dataset.columns[-1]] = dataset[dataset.columns[-1]].str.decode('utf8')
    dataset = dataset.dropna()  
    return dataset

def compare_accuracy_n_features(inputs, outputs, fold):

    test_accuracies = []
    train_accuracies = []
    for n_features in (1, 3, 5 ,7):
        inputs_new = feature_selection.SelectKBest(feature_selection.mutual_info_classif, k = n_features).fit_transform(inputs, outputs)
        results = model_selection.cross_validate(estimator = tree.DecisionTreeClassifier(criterion = "entropy"), X = inputs_new, y = outputs, scoring = "accuracy", cv = fold, return_train_score = True)
        test_accuracies.append(np.mean(results["test_score"]))
        train_accuracies.append(np.mean(results["train_score"]))
    
    fig, ax = plt.subplots()
    ax.plot([1, 3, 5, 7], test_accuracies, label = "test accuracies")
    ax.plot([1, 3, 5, 7], train_accuracies, label = "train accuracies")
    ax.legend(loc = "best")
    ax.set_xticks([1,3,5,7])

    plt.savefig("output/accuracy_n_features.pdf")

def compare_accuracy_n_depths(inputs, outputs, fold):

    test_accuracies = []
    train_accuracies = []
    for depth in (1, 3, 5 ,7):
        results = model_selection.cross_validate(estimator = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = depth), X = inputs, y = outputs, scoring = "accuracy", cv = fold, return_train_score = True)
        test_accuracies.append(np.mean(results["test_score"]))
        train_accuracies.append(np.mean(results["train_score"]))
    
    fig, ax = plt.subplots()
    ax.plot([1, 3, 5, 7], test_accuracies, label = "test accuracies")
    ax.plot([1, 3, 5, 7], train_accuracies, label = "train accuracies")
    ax.legend(loc = "best")
    ax.set_xticks([1,3,5,7])

    plt.savefig("output/accuracy_n_depths.pdf")

def main():
    dataset = load_data("../data/breast.w.arff")
    inputs = dataset.iloc[:, :-1].to_numpy()
    outputs = dataset.iloc[:, [-1]].to_numpy().T.flatten()
    kf = model_selection.KFold(n_splits = 10, shuffle = True, random_state = 76)
    compare_accuracy_n_features(inputs, outputs, kf)
    compare_accuracy_n_depths(inputs, outputs, kf)


if __name__ == "__main__":
    main()