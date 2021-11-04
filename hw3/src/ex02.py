import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import neural_network, model_selection, preprocessing, metrics
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

def load_data(filename):
    dataset = arff.loadarff(filename)
    dataset = pd.DataFrame(dataset[0])
    str_columns = [col for col in dataset.columns if dataset[col].dtype == "object"]
    dataset[str_columns] = dataset[str_columns].apply(lambda x: x.str.decode('utf8'))
    dataset = dataset.dropna()  
    return dataset

#  MLP with l2 regularization, RELU activation function, 2 hidden layers of size 3,2 and remaining default parameters.
def mlp_predict(mlp_model ,inputs, outputs, folds, early_stopping, alpha):
    clf = mlp_model(activation = 'relu',    hidden_layer_sizes = (3, 2), \
                                            random_state = 76, \
                                            early_stopping = early_stopping, \
                                            alpha = alpha, \
                                            max_iter = 1500)
    return model_selection.cross_val_predict(clf, inputs, outputs, cv = folds)

# Plots the confusion matrix for a given MLP with/without early stopping
def mlp_conf_matrix(inputs, outputs, folds, early_stopping):
    outputs_pred = mlp_predict(neural_network.MLPClassifier, inputs, outputs, folds, early_stopping, 1)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(outputs, outputs_pred)
    disp.ax_.set(title=f'Confusion Matrix {"(with early stopping)" if early_stopping else "(without early stopping)"}')
    plt.savefig(f"output/mlp_conf_matrix_{early_stopping}.pdf")

# Returns the residues of the distribution with/without regularization
def residue_dist_bp(inputs, outputs, folds):
    res_regularized = outputs - mlp_predict(neural_network.MLPRegressor, inputs, outputs, folds, True, 4)
    res_nonregularized = outputs - mlp_predict(neural_network.MLPRegressor, inputs, outputs, folds, True, 0)
    #np.savez("res.txt", res_regularized, res_nonregularized)

    #npzfile = np.load("res.txt.npz")
    #res_regularized = npzfile["arr_0"]
    #res_nonregularized = npzfile["arr_1"]

    fig, ax = plt.subplots()
    bp = ax.boxplot([res_regularized, res_nonregularized], vert = False, flierprops={'marker': 'o', 'markersize': 1})

    ax.set_yticklabels(['without regularization','with regularization \n ($\\alpha = 4$)'])
    ax.tick_params(axis = u'y', length = 0)
    fig.set_size_inches(12, 6)
    
    plt.savefig("output/residue_boxplot.pdf")
   

def main():
    kf = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 0)

    # 2) =====
    breast_data = load_data("../data/breast.w.arff")
    inputs_breast = breast_data.iloc[:, :-1].to_numpy()
    outputs_breast = breast_data.iloc[:, [-1]].to_numpy().T.flatten()
    mlp_conf_matrix(inputs_breast, outputs_breast, kf, True)
    mlp_conf_matrix(inputs_breast, outputs_breast, kf, False)

    # 3) =====
    kin_data = load_data("../data/kin8nm.arff")
    inputs_kin = kin_data.iloc[:, :-1].to_numpy()
    outputs_kin = kin_data.iloc[:, [-1]].to_numpy().T.flatten()
    residue_dist_bp(inputs_kin, outputs_kin, kf)
    
if __name__ == "__main__":
    main()