from joblib.logger import PrintTime
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import cluster, metrics, feature_selection
import matplotlib.pyplot as plt
#plt.rcParams["text.usetex"] = True

def load_data(filename):
    dataset = arff.loadarff(filename)
    dataset = pd.DataFrame(dataset[0])
    str_columns = [col for col in dataset.columns if dataset[col].dtype == "object"]
    dataset[str_columns] = dataset[str_columns].apply(lambda x: x.str.decode('utf8'))
    dataset = dataset.dropna()  
    return dataset

def compute_ecr(labels, outputs, k):
    benign_count = [0] * k
    malign_count = [0] * k

    for i in range(0, k):
        for j in range(0, len(labels)):
            if labels[j] == i:
                if outputs[j] == "benign":
                    benign_count[i]+=1
                else:
                    malign_count[i]+=1
    
    aux = np.absolute(np.array(benign_count) - np.array(malign_count))
    return (1/k) * (np.sum(aux))

### MAIN ### 
data = load_data("D:\\apre_homeworks\\hw4\\data\\breast.w.arff")
inputs = data.iloc[:, :-1].to_numpy()
outputs = data.iloc[:, [-1]].to_numpy().T.flatten()
kmeans_2 = cluster.KMeans(n_clusters=2, random_state=76).fit(inputs)
kmeans_3 = cluster.KMeans(n_clusters=3, random_state=76).fit(inputs)

# 4) a.
print(f"ECR k = 2: {compute_ecr(kmeans_2.labels_, outputs, 2)}")
print(f"ECR k = 3: {compute_ecr(kmeans_3.labels_, outputs, 3)}")
print(" --- ")

# 4) b.
print(f"Silhouette score k = 2: {round(metrics.silhouette_score(inputs, kmeans_2.labels_), 5)}")
print(f"Silhouette score k = 3: {round(metrics.silhouette_score(inputs, kmeans_3.labels_), 5)}")

# 5)
kBest = feature_selection.SelectKBest(feature_selection.mutual_info_classif, k = 2).fit(inputs, outputs)
inputs_top_features = inputs[:, kBest.get_support(indices=True)]
kmeans_3 = cluster.KMeans(n_clusters=3, random_state=76).fit(inputs_top_features)
plt.scatter(inputs_top_features[:,0], inputs_top_features[:,1], c=kmeans_3.labels_, cmap='rainbow')
plt.scatter(kmeans_3.cluster_centers_[:,0] ,kmeans_3.cluster_centers_[:,1], color='k')
plt.show()
