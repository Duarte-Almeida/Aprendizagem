import numpy as np
from termcolor import cprint

def print_matrix(matrix):
    for i in range(matrix.shape[0]):
        string = ""
        for j in range(matrix.shape[1]):
            string += f" {np.round(matrix[i][j], 4)} &"
        string = string[:-1] + "\\\\"
        print(string)


# feature mapping
def phi(x):
    phi = np.zeros(4)
    for i in range(4):
        phi[i] = np.linalg.norm(x) ** i
    return phi

def read_data(filename):
    dataset = np.loadtxt(filename, delimiter=",")
    return dataset

def linear_regression(inputs, outputs):
    # Φ
    Phi = np.apply_along_axis(phi, 1, inputs)
    cprint('Φ = ', 'red')
    print_matrix(Phi)
    
    # Φ^T · Φ
    phi_T_phi = np.matmul(Phi.T, Phi)
    cprint('Φ^T · Φ = ', 'red')
    print_matrix(phi_T_phi)

    # (Φ^T · Φ)^-1
    inverse = np.linalg.inv(phi_T_phi)
    cprint('(Φ^T · Φ)^-1 = ', 'red')
    print_matrix(inverse)

    # (Φ^T · Φ)^-1 · Φ^T
    pseudo_inverse = np.matmul(inverse, Phi.T)
    cprint('(Φ^T · Φ)^-1 · Φ^T = ', 'red')
    print_matrix(pseudo_inverse)

    # W = (Φ^T · Φ)^-1 · Φ^T · z
    weights = np.matmul(pseudo_inverse, outputs)
    cprint('W = (Φ^T · Φ)^-1 · Φ^T · z = ', 'green')
    print_matrix(weights)

    return weights

def get_rmse(inputs, outputs, weights):
    predictions = np.matmul(np.apply_along_axis(phi, 1, inputs), weights)
    cprint('Φ = ', 'green')
    print_matrix(np.apply_along_axis(phi, 1, inputs))
    cprint('Ẑ = ', 'green')
    print_matrix(predictions)
    rmse = np.sqrt((np.linalg.norm(predictions - outputs) ** 2) / outputs.shape[0])
    cprint(f'RMSE = {rmse}', 'green', attrs=['bold'])

#def categorize_dataset(dataset):
#    median = np.median(dataset[:, [3]])
#    dataset[:, [3]] = np.where(dataset[:, [3]] < median, 0, 1)
#    dataset[:, [4]] = np.where(dataset[:, [4]] < 4, 0, 1)
#
#    return dataset

#def compute_decision_tree(dataset):
#    print("")
    
def main():
    dataset = read_data("../data/data.csv")
    train = dataset[:8, :]
    test = dataset[8:, :]
    weights = linear_regression(inputs = train[: , :-1], outputs = train[: , -1:])
    get_rmse(inputs = test[:, :-1], outputs = test[:, -1:], weights = weights)
    #new_dataset = categorize_dataset(dataset)
    #compute_decision_tree(new_dataset)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()