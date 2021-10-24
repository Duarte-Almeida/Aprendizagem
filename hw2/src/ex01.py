import numpy as np
from termcolor import cprint

# feature mapping
def phi(x):
    phi = np.zeros(4)
    for i in range(4):
        phi[i] = np.linalg.norm(x) ** i
    return phi

def read_data(filename):
    dataset = np.loadtxt(filename, delimiter=",")
    dataset = np.concatenate((np.ones((dataset.shape[0], 1)), dataset), axis = 1)
    return dataset

def linear_regression(inputs, outputs):
    # Φ
    Phi = np.apply_along_axis(phi, 1, inputs)
    cprint('Φ = ', 'red')
    print(Phi)
    
    # Φ^T · Φ
    phi_T_phi = np.matmul(Phi.T, Phi)
    cprint('Φ^T · Φ = ', 'red')
    print(phi_T_phi)

    # (Φ^T · Φ)^-1
    inverse = np.linalg.inv(phi_T_phi)
    cprint('(Φ^T · Φ)^-1 = ', 'red')
    print(inverse)

    # (Φ^T · Φ)^-1 · Φ^T
    pseudo_inverse = np.matmul(inverse, Phi.T)
    cprint('(Φ^T · Φ)^-1 · Φ^T = ', 'red')
    print(pseudo_inverse)

    # W = (Φ^T · Φ)^-1 · Φ^T · z
    weights = np.matmul(pseudo_inverse, outputs)
    cprint('W = (Φ^T · Φ)^-1 · Φ^T · z = ', 'green')
    print(weights)

    return weights

def get_rmse(inputs, outputs, weights):
    predictions = np.matmul(np.apply_along_axis(phi, 1, inputs), weights)
    cprint('Ẑ = ', 'green')
    rmse = np.sqrt((np.linalg.norm(predictions - outputs) ** 2) / outputs.shape[0])
    cprint(f'RMSE = {rmse}', 'green', attrs=['bold'])

def categorize_dataset(dataset):
    median = np.median(dataset[:, [3]])
    dataset[:, [3]] = np.where(dataset[:, [3]] < median, 0, 1)
    dataset[:, [4]] = np.where(dataset[:, [4]] < 4, 0, 1)

    return dataset

def compute_decision_tree(dataset):
    print("")
    
def main():
    dataset = read_data("../data/data.csv")
    train = dataset[:, :8]
    test = dataset[:, 8:]
    weights = linear_regression(inputs = train[: , :-1], outputs = train[: , -1:])
    get_rmse(inputs = test[:, :-1], outputs = test[:, -1:], weights = weights)
    new_dataset = categorize_dataset(dataset)
    compute_decision_tree(new_dataset)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()