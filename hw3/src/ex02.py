import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import neural_network, model_selection, preprocessing, feature_selection, metrics
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

def load_data(filename):
    dataset = arff.loadarff(filename)
    dataset = pd.DataFrame(dataset[0])
    # REVIEW: this decode breaks for kin dataset but is necessary for breast_data
    try:
        dataset[dataset.columns[-1]] = dataset[dataset.columns[-1]].str.decode('utf8')
    except AttributeError:
        # kin will pass this
        pass
    dataset = dataset.dropna()  
    return dataset

#  MLP with l2 regularization, RELU activation function, 2 hidden layers of size 3,2 and remaining default parameters.
def mlp_predict(type,inputs, outputs, folds, early_stopping, alpha):
    if type == "classifier":
        clf = neural_network.MLPClassifier(activation = 'relu', solver = 'sgd', \
                                            hidden_layer_sizes = (3, 2), \
                                            random_state = 76, \
                                            early_stopping = early_stopping, \
                                            alpha=alpha, \
                                            max_iter=1500)
    elif type == "regressor":
        clf = neural_network.MLPRegressor(activation = 'relu', solver = 'sgd', \
                                        hidden_layer_sizes = (3, 2), \
                                        random_state = 76, \
                                        early_stopping = early_stopping, \
                                        alpha=alpha, \
                                        max_iter=1500)
    return model_selection.cross_val_predict(clf, inputs, outputs, cv=folds)

# Plots the confusion matrix for a given MLP with/without early stopping
def mlp_conf_matrix(inputs, outputs, folds, early_stopping):
    outputs_pred = mlp_predict("classifier", inputs, outputs, folds, early_stopping, 1)
    conf_mat = metrics.confusion_matrix(outputs, outputs_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Benign','Malign']).plot()
    disp.ax_.set(xlabel='Predicted', ylabel='True', \
        title=f'Confusion Matrix {"(with early stopping)" if early_stopping else "(without early stopping)"}')


# Plots the distribution of the residues using boxplots with/without regularization
def residue_dist_bp(inputs, outputs, folds, regularization):
    return outputs - mlp_predict("regressor", inputs, outputs, folds, True, 1 if regularization else 1e-5)


def main():
    kf = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 0)

    # 2) =====
    #breast_data = load_data("../data/breast.w.arff")
    #inputs_breast = breast_data.iloc[:, :-1].to_numpy()
    #outputs_breast = breast_data.iloc[:, [-1]].to_numpy().T.flatten()
    #mlp_conf_matrix(inputs_breast, outputs_breast, kf, True)
    #mlp_conf_matrix(inputs_breast, outputs_breast, kf, False)

    # 3) =====
    kin_data = load_data("../data/kin8nm.arff")
    inputs_kin = kin_data.iloc[:, :-1].to_numpy()
    outputs_kin = kin_data.iloc[:, [-1]].to_numpy().T.flatten()
    residue_bp1 = residue_dist_bp(inputs_kin, outputs_kin, kf, True)
    residue_bp2 = residue_dist_bp(inputs_kin, outputs_kin, kf, False)
    residues = np.column_stack((residue_bp1, residue_bp2))
    
    fig, disp = plt.subplots()
    disp.set_title('Multiple Samples with Different sizes')
    disp.boxplot(residues, vert=False)
    disp.set_axisbelow(True)
    disp.set_yticklabels(['with regularization','without regularization'])
    disp.set_title('Comparison of residues in the presence and absence of regularization')
    plt.show()

if __name__ == "__main__":
    main()