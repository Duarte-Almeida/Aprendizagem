import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import feature_selection, model_selection, tree
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

def load_data(filename):
    dataset = arff.loadarff(filename)
    dataset = pd.DataFrame(dataset[0])
    dataset[dataset.columns[-1]] = dataset[dataset.columns[-1]].str.decode('utf8')
    dataset = dataset.dropna()  
    return dataset

def compare_accuracy(inputs, outputs, folds, mode):
    test_accuracies = []
    train_accuracies = []
    for param in (1, 3, 5 ,9):
        if mode == "features":
            inputs_new = feature_selection.SelectKBest(feature_selection.mutual_info_classif, \
                                                       k = param).fit_transform(inputs, outputs)
            estimator = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 76)
            label = "number\:of\:features"
        elif mode == "tree_depths":
            inputs_new = inputs
            estimator = tree.DecisionTreeClassifier(criterion = "entropy", \
                                                    max_depth = param, random_state = 76)
            label = "tree\:depth"
        results = model_selection.cross_validate(estimator = estimator, \
                                                 X = inputs_new, y = outputs, \
                                                 scoring = "accuracy", cv = folds, \
                                                 return_train_score = True)
        test_accuracies.append(np.mean(results["test_score"]))
        train_accuracies.append(np.mean(results["train_score"]))
    fig, ax = plt.subplots()
    ax.plot([1, 3, 5, 9], test_accuracies, label = "test accuracies")
    ax.plot([1, 3, 5, 9], train_accuracies, label = "train accuracies")
    ax.set_yticks(test_accuracies + train_accuracies, minor=True)
    ax.set_xticks([1,3,5,7,9])
    ax.yaxis.grid(True, which='minor', alpha = 0.5, linestyle = "dashed")
    ax.legend(loc = "best")
    ax.set_xlabel(label)
    ax.set_ylabel("accuracy")
    plt.savefig(f"output/accuracy_n_{mode}.pdf")

dataset = load_data("../data/breast.w.arff")
inputs = dataset.iloc[:, :-1].to_numpy()
outputs = dataset.iloc[:, [-1]].to_numpy().T.flatten()
kf = model_selection.KFold(n_splits = 10, shuffle = True, random_state = 76)
compare_accuracy(inputs, outputs, kf, "features")
compare_accuracy(inputs, outputs, kf, "tree_depths")


