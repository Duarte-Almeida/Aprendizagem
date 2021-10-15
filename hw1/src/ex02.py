from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

# loads .arff dataset into a dataframee
def load_data(filename):

    dataset = arff.loadarff(filename)
    dataset = pd.DataFrame(dataset[0])

    # remove 'b' letter that was in the Class column
    dataset[dataset.columns[-1]] = dataset[dataset.columns[-1]].str.decode('utf8') 

    return dataset

def plot_histograms(dataset):

    bins = np.linspace(0, 10, 4)
    print(bins)
    for feature in dataset.columns[:-1]:
        label = feature.replace("_", "\:")
        _ = plt.hist(dataset[dataset["Class"] == "benign"][feature], bins = bins, density = True, label = f"$p(\mathrm{{{label}|\: Class = benign}})$", alpha = 0.5)
        _ = plt.hist(dataset[dataset["Class"] == "malignant"][feature], bins = bins, density = True, label = f"$p(\mathrm{{{label}|\: Class = malignant}})$", alpha = 0.5)
        plt.legend(loc='best')
        plt.xticks(bins)
        plt.show()
       
        
def main():

    dataset = load_data("../data/breast.w.arff")
    plot_histograms(dataset)
    
    
if __name__ == "__main__":
    main()
   