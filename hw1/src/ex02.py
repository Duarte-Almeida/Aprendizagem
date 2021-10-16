import numpy as np
from scipy.io import arff
from scipy import stats
from sklearn import model_selection, metrics, neighbors, naive_bayes
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

GROUP_NUMBER = 76

# loads .arff dataset into a dataframe
def load_data(filename):

    dataset = arff.loadarff(filename)
    dataset = pd.DataFrame(dataset[0])

    # remove 'b' letter that was in the Class column and remove instances with missing features
    dataset[dataset.columns[-1]] = dataset[dataset.columns[-1]].str.decode('utf8') 
    dataset = dataset.dropna()
    
    return dataset

# plot class conditional distributions
# each plot depicts p(yi | Class = beningn) and p(yi | Class = malignant) for each feature yi
def plot_histograms(dataset):
    bins = np.linspace(1, 11, 11)           # create bins from 1 to 10
    for feature in dataset.columns[:-1]:
        label = feature.replace("_", "\:")
        _ = plt.hist(dataset[dataset["Class"] == "benign"][feature], bins = bins, density = True, label = f"$p(\mathrm{{{label}|\: Class = benign}})$", alpha = 0.5, rwidth = 0.5, align = "left", color = "green")
        _ = plt.hist(dataset[dataset["Class"] == "malignant"][feature], bins = bins, density = True, label = f"$p(\mathrm{{{label}|\: Class = malignant}})$", alpha = 0.5, rwidth = 0.5, align = "left", color = "red")
        plt.legend(loc='best')
        plt.xticks(np.linspace(1, 10, 10))
        plt.xlim(right = 11)                # ignore xtick with the number 11
        plt.savefig(f"output/{feature}.pdf")
        plt.clf()

# perform 10 fold cross validation on dataset using a kNN classifier
# and acess average accuracy for k = 3, 5, 7
def kNN_cross_validation(dataset):

    kf = model_selection.KFold(n_splits = 10, shuffle = True, random_state = GROUP_NUMBER)
    inputs = dataset.iloc[:, : -1].values
    outputs = dataset[dataset.columns[-1]].values

    fig, ax = plt.subplots()
    test_errors = {}
    for k in (3, 5, 7):
        kNN = neighbors.KNeighborsClassifier(n_neighbors = k)
        cv_results = model_selection.cross_validate(estimator = kNN, X = inputs, y = outputs, scoring = "accuracy", cv = kf)
        test_errors[k] = cv_results["test_score"]
        print(f"{k}NN accuracy average and variance: {np.mean(test_errors[k])} / {np.var(test_errors[k], ddof = 1)}")
     
    bplot = ax.boxplot(test_errors.values())
    ax.set_xticklabels(["{}-NN".format(k) for k in test_errors.keys()])
    plt.savefig("output/kNN_performances.pdf")

# test the hypothesis that kNN is statistically superior to multinomial Naive Bayes
# using a one-sided t test
def test_kNN_NBayes(dataset):

    kf = model_selection.KFold(n_splits = 10, shuffle = True, random_state = GROUP_NUMBER)
    inputs = dataset.iloc[:, : -1].values
    outputs = dataset[dataset.columns[-1]].values

    kNN = neighbors.KNeighborsClassifier(n_neighbors = 3)
    NBayes = naive_bayes.MultinomialNB()

    accuracies_kNN = model_selection.cross_validate(estimator = kNN, X = inputs, y = outputs, scoring = "accuracy", cv = kf)["test_score"]
    accuracies_NB = model_selection.cross_validate(estimator = NBayes, X = inputs, y = outputs, scoring = "accuracy", cv = kf)["test_score"]

    pvalue = stats.ttest_rel(accuracies_kNN, accuracies_NB, alternative = "greater").pvalue
    print(f"p-value:{pvalue}")
    
def main():
    dataset = load_data("../data/breast.w.arff")
    plot_histograms(dataset)
    kNN_cross_validation(dataset)
    test_kNN_NBayes(dataset)

if __name__ == "__main__":
    main()
   