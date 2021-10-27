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

def compare_accuracy_n_features(inputs, outputs):

    test_accuracies = []
    train_accuracies = []
    kf = model_selection.KFold(n_splits = 10, shuffle = True, random_state = 76)
    for n_features in (1, 3, 5 ,9):
        inputs_new = feature_selection.SelectKBest(feature_selection.mutual_info_classif, k = n_features).fit_transform(inputs, outputs)
        results = model_selection.cross_validate(estimator = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 76), X = inputs_new, y = outputs, scoring = "accuracy", cv = kf, return_train_score = True)
        test_accuracies.append(np.mean(results["test_score"]))
        train_accuracies.append(np.mean(results["train_score"]))
    
    print(f"training accuracies:{train_accuracies} \ntesting accuracies:{test_accuracies}")
    fig, ax = plt.subplots()
    ax.plot([1, 3, 5, 9], test_accuracies, label = "test accuracies")
    ax.plot([1, 3, 5, 9], train_accuracies, label = "train accuracies")
    ax.set_yticks(test_accuracies + train_accuracies, minor=True)
    ax.yaxis.grid(True, which='minor', alpha = 0.5, linestyle = "dashed")
    ax.legend(loc = "best")
    ax.legend(loc = "best")
    ax.set_xticks([1,3,5,7,9])

    plt.savefig("output/accuracy_n_features.pdf")

def compare_accuracy_n_depths(inputs, outputs):

    test_accuracies = []
    train_accuracies = []
    kf = model_selection.KFold(n_splits = 10, shuffle = True, random_state = 76)
    for depth in (1, 3, 5 ,9):
        results = model_selection.cross_validate(estimator = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = depth, random_state = 76), X = inputs, y = outputs, scoring = "accuracy", cv = kf, return_train_score = True)
        test_accuracies.append(np.mean(results["test_score"]))
        train_accuracies.append(np.mean(results["train_score"]))
    print(f"training accuracies:{train_accuracies} \ntesting accuracies:{test_accuracies}")
    fig, ax = plt.subplots()
    ax.plot([1, 3, 5, 9], test_accuracies, label = "test accuracies")
    ax.plot([1, 3, 5, 9], train_accuracies, label = "train accuracies")
    ax.set_yticks(test_accuracies + train_accuracies, minor=True)
    ax.yaxis.grid(True, which='minor', alpha = 0.5, linestyle = "dashed")
    ax.legend(loc = "best")
    ax.set_xticks([1,3,5,7,9])

    plt.savefig("output/accuracy_n_depths.pdf")

def main():
    dataset = load_data("../data/breast.w.arff")
    inputs = dataset.iloc[:, :-1].to_numpy()
    outputs = dataset.iloc[:, [-1]].to_numpy().T.flatten()
    compare_accuracy_n_features(inputs, outputs)
    compare_accuracy_n_depths(inputs, outputs)


if __name__ == "__main__":
    main()