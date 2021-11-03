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

def do_some_shit():
    # do some shit

def do_some_other_shit():
    # do some other shit

def main():
    breast_data = load_data("../data/breast.w.arff")
    do_some_shit()
    kin_data = load_data("../data/kin8nm.w.arff")
    do_some_other_shit()


if __name__ == "__main__":
    main()