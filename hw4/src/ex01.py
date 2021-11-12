import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import neural_network, model_selection, metrics
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

def load_data(filename):
    dataset = arff.loadarff(filename)
    dataset = pd.DataFrame(dataset[0])
    str_columns = [col for col in dataset.columns if dataset[col].dtype == "object"]
    dataset[str_columns] = dataset[str_columns].apply(lambda x: x.str.decode('utf8'))
    dataset = dataset.dropna()  
    return dataset

   
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