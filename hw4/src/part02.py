import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.stats import mode
from sklearn import cluster, metrics, feature_selection
import matplotlib.pyplot as plt
from collections import Counter
plt.rcParams["text.usetex"] = True

def load_data(filename):
    dataset = arff.loadarff(filename)
    dataset = pd.DataFrame(dataset[0])
    str_columns = [col for col in dataset.columns if dataset[col].dtype == "object"]
    dataset[str_columns] = dataset[str_columns].apply(lambda x: x.str.decode('utf8'))
    dataset = dataset.dropna()  
    return dataset

# ECR is the mean over all cluster of cluster_size - label_mode_in_each_cluster
def compute_ecr(labels, outputs):

    cluster_indexes = np.unique(labels)
    ecr_measure = 0
    
    for index in cluster_indexes:
        cluster = outputs[labels == index]  # get instances in each cluster
        c = cluster.shape[0]
        phi = np.amax(np.unique(cluster, return_counts = True)[1]) # get count of majority label in cluster
        ecr_measure += c - phi
    
    return ecr_measure / cluster_indexes.shape[0]
                             
data = load_data("../data/breast.w.arff")
inputs = data.iloc[:, :-1].to_numpy()
outputs = data.iloc[:, [-1]].to_numpy().T.flatten()
labels_2means = cluster.KMeans(n_clusters = 2, random_state = 76).fit_predict(inputs)
labels_3means = cluster.KMeans(n_clusters = 3, random_state = 76).fit_predict(inputs)

print(f"ECR k = 2: {compute_ecr(labels_2means, outputs)}")
print(f"ECR k = 3: {compute_ecr(labels_3means, outputs)}")
print(" --- ")

print(f"Silhouette score k = 2: {round(metrics.silhouette_score(inputs, labels_2means), 5)}")
print(f"Silhouette score k = 3: {round(metrics.silhouette_score(inputs, labels_3means), 5)}")

mutual_info_3best = feature_selection.SelectKBest(feature_selection.mutual_info_classif, k = 2)
inputs_top_features = mutual_info_3best.fit_transform(inputs, outputs)
extracted_features = data.iloc[:, :-1].columns[mutual_info_3best.get_support()].values
clusters_3means = cluster.KMeans(n_clusters = 3, random_state = 76).fit(inputs_top_features)

fig, ax = plt.subplots()
for index, c in zip(np.unique(clusters_3means.labels_), ("indianred", "gold", "darkslategrey")): #TODO: select another color palette
    #weights = [2*i for i in Counter(inputs_top_features[:, 0][clusters_3means.labels_ == index]).values() for j in range(i)]
    ax.scatter(inputs_top_features[:, 0][clusters_3means.labels_ == index], \
                inputs_top_features[:, 1][clusters_3means.labels_ == index], color = c)

#plt.scatter(inputs_top_features[:, 0], inputs_top_features[:, 1], c = clusters_3means.labels_, cmap='cividis') #TODO: check this

ax.scatter(clusters_3means.cluster_centers_[:, 0] , clusters_3means.cluster_centers_[:, 1], color = "k", marker = "+", linewidths = 0.5, s = 150) 
ax.set_xlabel(extracted_features[0].replace("_", "\;"))
ax.set_ylabel(extracted_features[1].replace("_", "\;"))
plt.savefig("output/3means.pdf")