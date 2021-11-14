import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import cluster, metrics, feature_selection
import matplotlib.pyplot as plt
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
clusters_2means = cluster.KMeans(n_clusters = 2, random_state = 76).fit(inputs)
clusters_3means = cluster.KMeans(n_clusters = 3, random_state = 76).fit(inputs)

print(f"ECR k = 2: {compute_ecr(clusters_2means.labels_, outputs)}")
print(f"ECR k = 3: {compute_ecr(clusters_3means.labels_, outputs)}")
print(f"Silhouette score k = 2: {round(metrics.silhouette_score(inputs, clusters_2means.labels_), 5)}")
print(f"Silhouette score k = 3: {round(metrics.silhouette_score(inputs, clusters_3means.labels_), 5)}")

mutual_info_2best = feature_selection.SelectKBest(feature_selection.mutual_info_classif, k = 2)
inputs_top_features = mutual_info_2best.fit_transform(inputs, outputs)
extracted_features = data.iloc[:, :-1].columns[mutual_info_2best.get_support()].values
cluster_centers = clusters_3means.cluster_centers_[:, mutual_info_2best.get_support()]


fig, ax = plt.subplots()
for index, c in zip(np.unique(clusters_3means.labels_), ("r", "g", "b")): 
    ax.scatter(inputs_top_features[:, 0][clusters_3means.labels_ == index], \
               inputs_top_features[:, 1][clusters_3means.labels_ == index], \
               color = c, alpha = 0.2, label = "$\\mathrm{{Cluster\;{}}}$".format(index))
    ax.scatter(cluster_centers[index, 0], cluster_centers[index, 1], \
               color = c, marker = "+", linewidths = 0.5, s = 150, label = "$\\mathrm{{Cluster\;{}\;centroid}}$".format(index))

plt.legend(loc = "best", prop = {'size': 9}, markerscale = 0.75) 
ax.set_xlabel("$\mathrm{{ {} }}$".format(extracted_features[0].replace("_", "\;")))
ax.set_ylabel("$\mathrm{{ {} }}$".format(extracted_features[1].replace("_", "\;")))
plt.savefig("output/3means.pdf")