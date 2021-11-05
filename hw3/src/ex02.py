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

def mlp_predict(mlp_model ,inputs, outputs, folds, early_stopping, alpha):
    clf = mlp_model(activation = 'relu',    hidden_layer_sizes = (3, 2), early_stopping = early_stopping, \
                                            alpha = alpha, random_state = 76, max_iter = 1500)
    return model_selection.cross_val_predict(clf, inputs, outputs, cv = folds)

def mlp_conf_matrix(inputs, outputs, folds, early_stopping):
    outputs_pred = mlp_predict(neural_network.MLPClassifier, inputs, outputs, folds, early_stopping, 0.4)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(outputs, outputs_pred)
    disp.ax_.set(title=f'Confusion Matrix (with early stopping)' if early_stopping else \
                       f'Confusion Matrix (without early stopping)')
    plt.savefig(f"output/mlp_conf_matrix_{early_stopping}.pdf")

def residue_dist_bp(inputs, outputs, folds):
    res_regularized = outputs - mlp_predict(neural_network.MLPRegressor, inputs, outputs, folds, True, 4)
    res_nonregularized = outputs - mlp_predict(neural_network.MLPRegressor, inputs, outputs, folds, True, 0)
    fig, ax = plt.subplots()
    bp = ax.boxplot([res_regularized, res_nonregularized], \
                    vert = False, flierprops={'marker': 'o', 'markersize': 2})
    ax.set_yticklabels(['without regularization','with $l_2$ regularization \n ($\\alpha = 4$)'])
    ax.tick_params(axis = u'y', length = 0)
    ax.set(title=f"Distribution of residues")
    fig.set_size_inches(12, 6)
    plt.savefig("output/residue_boxplot.pdf")
   
kf = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 0)

breast_data = load_data("../data/breast.w.arff")
inputs_breast = breast_data.iloc[:, :-1].to_numpy()
outputs_breast = breast_data.iloc[:, [-1]].to_numpy().T.flatten()
mlp_conf_matrix(inputs_breast, outputs_breast, kf, True)
mlp_conf_matrix(inputs_breast, outputs_breast, kf, False)

kin_data = load_data("../data/kin8nm.arff")
inputs_kin = kin_data.iloc[:, :-1].to_numpy()
outputs_kin = kin_data.iloc[:, [-1]].to_numpy().T.flatten()
residue_dist_bp(inputs_kin, outputs_kin, kf)
    
