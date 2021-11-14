import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

X = np.array([[2, 4], \
              [-1, -4], 
              [-1, 2], 
              [4, 0]])

Sigma = np.array([[[1, 0], \
                   [0, 1]], \
                  [[2, 0], \
                   [0, 2]]])

pi = np.array([0.7, 0.3])

def print_matrix(matrix):
    for i in range(matrix.shape[0]):
        str = ""
        for j in range(matrix.shape[1]):
            str += f" {round(matrix[i][j], 5)} &"
        str = str[:-1] + "\\\\"
        print(str)

def expectation(X, Sigma, priors):
    print("Expectation\n\n")
    Gama = np.zeros((X.shape[0], priors.shape[0]))
    for n in range(0, X.shape[0]):
        p_x = 0
        joint_probs = np.zeros(priors.shape[0])
        print(f"x_{n+1}\n")
        for k in range(0, priors.shape[0]):
            p_x_given_c = stats.multivariate_normal(mean = X[k].flatten(), cov = Sigma[k]).pdf(X[n])
            joint_probs[k] = p_x_given_c * priors[k]
            p_x += joint_probs[k]
            print(f"p(c = {k + 1}) = {round(priors[k], 5)}   p(x_{n + 1}|c = {k + 1}) = {round(p_x_given_c, 5)}  p(x_{n + 1}, c = {k + 1}) = {round(joint_probs[k], 5)}")   
        print(f"p(x = {n}) = {round(p_x, 5)}") 
        str=""
        for k in range(0, priors.shape[0]):
            Gama[n][k] = joint_probs[k] / p_x
            str += f"p(c = {k + 1}|x_{n+1}) = {round(Gama[n][k], 5)}   "
        print(str + "\n")
    
    return Gama


def maximization(X, Sigma, priors, Gama):

    print("Maximization\n")

    # Calculate new mu's
    normalized = np.apply_along_axis(lambda x: x / np.sum(x), 0, Gama)
    new_Mu = np.matmul(X.T, normalized)

    for k in range(0, new_Mu.shape[1]):
        print(f"mu_{k + 1} =")
        print_matrix(new_Mu[:, [k]])

    normalized_sqrt = np.apply_along_axis(lambda x: np.sqrt(x) / np.sqrt(np.sum(x)), 0, Gama)
    new_Sigma = np.zeros((priors.shape[0], X.shape[1], X.shape[1]))

    for k in range(0, priors.shape[0]):
        aux = normalized_sqrt[:, [k]].T * (X.T - new_Mu[:, [k]])
        new_Sigma[k] = np.matmul(aux, aux.T)
        print(f"Sigma_{k + 1} =")
        print_matrix(new_Sigma[k])

    new_pi = np.apply_along_axis(lambda x: np.sum(x) / x.shape[0], 0, Gama)
    
    for k in range(0, priors.shape[0]):
        print(f"pi_{k + 1} = {round(new_pi[k], 5)}")

    return new_Mu, new_Sigma, new_pi

def plot_clusters(X, new_Mu, new_Sigma, new_pi):
    fig, ax = plt.subplots()
    plt.scatter(X.T[0], X.T[1])

    x = np.arange(-8, 8, 0.1)
    y = np.arange(-8, 8, 0.1)
    X, Y = np.meshgrid(x, y)
    
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    for k in range(0, pi.shape[0]):
        ax.contour(X, Y, stats.multivariate_normal(mean = new_Mu[:, [k]].flatten(), cov = new_Sigma[k]).pdf(np.dstack((X, Y))), \
                                                   alpha = 0.5, cmap = cmaps[(2 * k) % len(cmaps)])
    
    fig.savefig("output/EM_clusters.pdf")

def compute_silhouette(X, Gama):
    
    cluster = np.apply_along_axis(lambda x: np.argmax(x), 1, Gama)
    cluster_dict = {k: [] for k in range(0, pi.shape[0])}
    for n in range(0, cluster.shape[0]):
        cluster_dict[cluster[n]].append(n)
    
    print("Silhouette calculation:\n")
    silhouettes = np.empty_like(pi, dtype = float)

    for c in cluster_dict.keys():
        print(f"For cluster c_{c + 1}: \n")
        if len(cluster_dict[c]) == 1:
            silhouettes[c] = 1                                  # TODO: send mail
            print(f"Silhouette(c_{c + 1}): {round(silhouettes[c], 5)}\n")
            continue
        intra_cluster_silhouettes = []
        for n in cluster_dict[c]:
            print(f"For x_{n + 1}: \n")
            distances = np.zeros_like(pi)
            for index in cluster_dict.keys():
                cluster_distances = []
                for m in (x for x in cluster_dict[index] if x != n):
                    cluster_distances.append(np.linalg.norm(X[n] - X[m]))
                    print(f"||x_{n + 1} - x_{m + 1}|| = {round(cluster_distances[-1], 5)}")
                distances[index] = np.mean(np.array(cluster_distances))
                print(f"Mean distance to cluster {index + 1}: {round(distances[index], 5)}\n")

            a = distances[cluster[n]]
            b = np.amin(np.delete(distances, cluster[n]))
            intra_cluster_silhouettes.append(1.0 - (a / b))
            print(f"Silhouette(x_{n + 1}): {round(intra_cluster_silhouettes[-1], 5)}\n")
        silhouettes[c] = np.mean(np.array(intra_cluster_silhouettes))
        print(f"Silhouette(c_{c + 1}): {round(silhouettes[c], 5)}\n")

    print(f"Cluster silhouette: {round(np.mean(silhouettes), 5)}")
    

def main():
    Gama = expectation(X, Sigma, pi)
    new_Mu, new_Sigma, new_pi = maximization(X, Sigma, pi, Gama)
    plot_clusters(X, new_Mu, new_Sigma, new_pi)
    compute_silhouette(X, Gama)


if __name__ == "__main__":
    main()