import numpy as np
from scipy.io import arff
from scipy import stats
from sklearn import model_selection, metrics, neighbors, naive_bayes, feature_selection
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

# loads .arff file to a dataset
def load_data(filename):

    dataset = arff.loadarff(filename)
    dataset = pd.DataFrame(dataset[0])

    # remove b's in "class" and instances with missing features
    dataset[dataset.columns[-1]] = dataset[dataset.columns[-1]].str.decode('utf8')
    dataset = dataset.dropna()  
    return dataset

# plots a 3 x 3 grid with class conditional distributions
def plot_grid(dataset):

    fig, axs = plt.subplots(3, 3)
    bins = np.linspace(1, 11, 11)           # create bins from 1 to 10

    for (index, feature) in zip([(i, j) for i in range(0, 3) for j in range(0, 3)], dataset.columns[:-1]):

        label = feature.replace("_", "\:")
        _ = axs[index[0], index[1]].hist(dataset[dataset["Class"] == "benign"][feature], bins = bins, density = True, label = f"$p(\mathrm{{{label}|\: Class = benign}})$", alpha = 0.5, rwidth = 0.5, align = "left", color = "green")
        _ = axs[index[0], index[1]].hist(dataset[dataset["Class"] == "malignant"][feature], bins = bins, density = True, label = f"$p(\mathrm{{{label}|\: Class = malignant}})$", alpha = 0.5, rwidth = 0.5, align = "left", color = "red")

        axs[index[0], index[1]].legend(loc='best', fontsize = 7.5)
        axs[index[0], index[1]].set_xticks(np.linspace(1, 10, 10))
        axs[index[0], index[1]].set_yticks(np.linspace(0, 1, 5))
        axs[index[0],index[1]].tick_params(axis = 'both', labelsize = 8)
        axs[index[0], index[1]].set_xlim(right = 11)                # ignore xtick with the number 11

    fig.set_size_inches(12, 10)
    plt.savefig(f"output/grid.jpg", dpi = 1200)

# perform 10-fold cross validation on k-NN (k = 3, 5, 7)
def kNN_cross_validation(inputs, outputs):

    kf = model_selection.KFold(n_splits = 10, shuffle = True, random_state = 76)
    fig, ax = plt.subplots()
    test_errors = {}

    for k in (3, 5, 7):
        test_errors[k] = model_selection.cross_validate(estimator = neighbors.KNeighborsClassifier(n_neighbors = k), X = inputs, y = outputs, scoring = "accuracy", cv = kf)["test_score"]
        print(f"{k}NN accuracy average and variance: {np.mean(test_errors[k])} / {np.var(test_errors[k], ddof = 1)}")

    bplot = ax.boxplot(test_errors.values())
    ax.set_xticklabels(["{}-NN".format(k) for k in test_errors.keys()])
    fig.set_size_inches(6, 2)
    plt.savefig("output/kNN_performances.png", dpi = 1200)

# t-test on equal performance vs kNN superiority relative to Naive Bayes
def test_kNN_NBayes(inputs, outputs):

    kf = model_selection.KFold(n_splits = 10, shuffle = True, random_state = 76)
    accuracies_kNN = model_selection.cross_validate(estimator = neighbors.KNeighborsClassifier(n_neighbors = 3), X = inputs, y = outputs, scoring = "accuracy", cv = kf)["test_score"]
    accuracies_NB = model_selection.cross_validate(estimator = naive_bayes.MultinomialNB(), X = inputs, y = outputs, scoring = "accuracy", cv = kf)["test_score"]
    result = stats.ttest_rel(accuracies_kNN, accuracies_NB, alternative = "greater")
    print(f"Statistic:{result.statistic} p-value:{result.pvalue}")
    
def main():
    dataset = load_data("../data/breast.w.arff")
    plot_grid(dataset)
    kNN_cross_validation(inputs = dataset.iloc[:, : -1].values,  outputs = dataset[dataset.columns[-1]].values)
    test_kNN_NBayes(inputs = dataset.iloc[:, : -1].values,  outputs = dataset[dataset.columns[-1]].values)

if __name__ == "__main__":
    main()