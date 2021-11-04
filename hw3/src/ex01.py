import numpy as np
from scipy.special import softmax
from termcolor import cprint

# make latex typesetting easier
def print_matrix(matrix):
    for i in range(len(matrix)):
        string = ""
        for j in range(len(matrix[i])):
            string += f" {round(matrix[i][j], 5)} &"
        string = string[:-1] + " \\\\"
        print(string)

# For ease of notation, we define W[0] = None since there is no layer -1
weights =[None, 
          np.array([[1., 1., 1., 1., 1.], 
                    [0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1.]]), 
          np.array([[1., 1., 1.],
                    [1., 1., 1.]]),
          np.array([[0., 0.],
                    [0., 0.]])]

# Make column vectors for biases to facilite printing (instead of numpy arrays)
biases_arrays = [np.array([1., 1., 1.]), np.array([1., 1.]), np.array([0., 0.])]
biases = [None] + [bias.reshape((bias.shape[0], 1)) for bias in biases_arrays]

def sqerror_dvt(target, output):
    return output - target 

def crentropy_dvt(target, output):
    return - target / output

# define derivatives in term of the function 

# tanh_line = 1 - tanh^2
def tanh_dvt(tanh):
    return np.vectorize(lambda el: 1 - el ** 2)(tanh)

# jacobian of phi(x) = (tanh(x1), ..., tanh(x2))
def tanh_gradient(tanh):
    return np.diagflat(tanh_dvt(tanh))

# derivative of phi(x) = softmax(x) 
def softmax_gradient(sf):
    jacobian = - np.matmul(sf.reshape((sf.shape[0], 1)), sf.reshape((1, sf.shape[0])))
    np.fill_diagonal(jacobian, sf * (1 - sf))
    return jacobian

# perform a stochastic backprop step on a neural net
# - input: input (vector)
# - target: target (vector)
# - weights & biases: initial parameters
# - error_fn_dvt: derivative of the error function (vector x vector -> scalar)
# - layer_act_fn: activation function applied to each hidden layer (vector -> vector)
# - layer_act_fn: activation function derivative evaluated at the activation instead of the net (vector -> vector)
# - output_act_fn: activation function applied to the output layer (vector -> vector)
# - output_act_fn:  function derivative evaluated at the activation instead of the net (vector -> vector
# - learning_rate: learning rate to be used in the SGD step
def backprop(input, target, weights, biases, error_fn_dvt, layer_act_fn, layer_act_fn_dvt, \
             output_act_fn, output_act_fn_dvt, learning_rate):

    # Initialize, nets, activations and deltas 
    net = [np.empty_like(bias) for bias in biases]
    activation = [np.empty_like(net) for net in net]
    delta = [np.empty_like(net) for net in net]
    weight_gradient = [np.empty_like(matrix) for matrix in weights]
    biases_gradient = [np.empty_like(bias) for bias in biases]
    new_weights= [np.empty_like(matrix) for matrix in weights]
    new_biases = [np.empty_like(bias) for bias in biases] 

    #set nets and activations in the first layer with input values 
    net[0] = activation[0] = input    
    
    # Forward propagation
    for i in range(1, len(weights) - 1):

        net[i] = np.matmul(weights[i], activation[i - 1]) + biases[i]
        activation[i] = layer_act_fn(net[i])

        #region
        cprint(f"\nLayer {i}", "blue")
        cprint(f"W^[{i}]: ", "green")
        print_matrix(weights[i])
        cprint(f"a^[{i - 1}]: ", "green")
        print_matrix(activation[i - 1])
        cprint(f"W^[{i}]a^[{i - 1}]", "green")
        print_matrix(np.matmul(weights[i], activation[i - 1]))
        cprint(f"b^[{i}]: ", "green")
        print_matrix(biases[i])
        cprint(f"net^[{i}]: ", "green")
        print_matrix(net[i])
        cprint(f"a^[{i}]: ", "green")
        print_matrix(activation[i])
        #endregion
    
    net[-1] = np.matmul(weights[-1], activation[-2]) + biases[-1]
    activation[-1] = output_act_fn(net[-1])

    #region
    cprint(f"\nLayer {len(weights) - 1}", "blue")
    cprint(f"W^[{len(weights) - 1}]: ", "green")
    print_matrix(weights[-1])
    cprint(f"a^[{len(weights) - 2}]: ", "green")
    print_matrix(activation[-2])
    cprint(f"W^[{len(weights) - 1}]a^[{len(weights) - 2}]", "green")
    print_matrix(np.matmul(weights[-1], activation[-2]))
    cprint(f"b^[{len(weights) - 1}]: ", "green")
    print_matrix(biases[len(weights) - 1])
    cprint(f"net^[{len(weights) - 1}]: ", "green")
    print_matrix(net[len(weights) - 1])
    cprint(f"a^[{len(weights) - 1}]: ", "green")
    print_matrix(activation[-1])
    #endregion

    # Backpropagation
    d_act_d_net = output_act_fn_dvt(activation[-1])
    d_E_d_act = error_fn_dvt(target, activation[-1])
    delta[-1] = np.matmul(d_act_d_net, d_E_d_act)
    weight_gradient[-1] = np.matmul(delta[-1], activation[-2].T)
    biases_gradient[-1] = delta[-1]
    
    #region
    # PRINTING SECTION 
    cprint(f"\nLayer {len(delta) - 1}: \n", "blue")
    cprint(f"Delta: ", "red")
    cprint(f"∇a_net: ", "green")
    print_matrix(d_act_d_net)
    cprint(f"∇E_a: ", "green")
    print_matrix(d_E_d_act)
    cprint(f"δ^[{len(delta) - 1}] =  ∇a_net ∇E_a", "green")
    print_matrix(delta[-1])

    cprint(f"\nWeights and biases: ", "red")
    cprint(f"a^[{len(delta) - 2}]^T:", "green")
    print_matrix(activation[-2])
    cprint(f"δ^[{len(delta) - 1}]:", "green")
    print_matrix(delta[-1])
    cprint(f"∇W = δ^[{len(delta) - 1}] a^[{len(delta) - 2}]^T: ", "green")
    print_matrix(weight_gradient[-1])
    cprint(f"∇b = δ^[{len(delta) - 1}]: ", "green")
    print_matrix(biases_gradient[-1])
    #endregion

    new_weights[-1] = weights[-1] - learning_rate * weight_gradient[-1]
    new_biases[-1] = biases[-1] - learning_rate * biases_gradient[-1]

    #region
    cprint(f"\nW^new^[{len(delta) - 1}]:", "magenta")
    print_matrix(new_weights[-1])
    cprint(f"b^new^[{len(delta) - 1}]:", "magenta")
    print_matrix(new_biases[-1])
    #endregion
    
    for i in range(len(weights) - 2, 0, -1):

        d_E_d_act = np.matmul(weights[i + 1].T, delta[i + 1])
        d_act_d_net = layer_act_fn_dvt(activation[i])
        delta[i] = layer_act_fn_dvt(activation[i]) * np.matmul(weights[i + 1].T, delta[i + 1])
        weight_gradient[i] = np.matmul(delta[i], activation[i - 1].T)
        biases_gradient[i] = delta[i]
        
        # region
        # PRINTING SECTION
        cprint(f"\nLayer {i}: \n", "blue")
        cprint(f"δ^[{i + 1}]:", "green")
        print_matrix(delta[i + 1])
        cprint(f"W^[{i + 1}]^T: ", "green")
        print_matrix(weights[i + 1].T)
        cprint(f"W^[{i + 1}]^T  δ^[{i + 1}]", "green")
        print_matrix(d_E_d_act)
        cprint(f"φ_line(a^{i}): ", "green")
        print_matrix(d_act_d_net)
        cprint(f"δ^[{i}] = φ_line(net^[{i + 1}]) o (W^[{i + 1}]^T δ^[{i + 1}]) :", "green")
        print_matrix(delta[i])

        cprint(f"\nWeights and biases: ", "red")
        cprint(f"δ^[{i}]  :", "green")
        print_matrix(delta[i])
        cprint(f"a^[{i - 1}]^T", "green")
        print_matrix(activation[i - 1])
        cprint(f"∇W = δ^[{i}] a^[{i - 1}]^T: ", "green")
        print_matrix(weight_gradient[i])
        cprint(f"∇b = δ^[{i}]: ", "green")
        print_matrix(biases_gradient[i])
        #endregion

        # Perform stochastic gradient descent
        new_weights[i] = weights[i] - learning_rate * weight_gradient[i]
        new_biases[i] = biases[i] - learning_rate * biases_gradient[i]
        
        #region
        cprint(f"\nW^new^[{i}]:", "magenta")
        print_matrix(new_weights[i])
        cprint(f"b^new^[{i}]:", "magenta")
        print_matrix(new_biases[i])
        #endregion
        

def main():
    #backprop(np.array([[1,1,1,1,1]]).T, np.array([[1, -1]]).T, weights, biases, \
    #         sqerror_dvt, np.tanh, tanh_dvt, np.tanh, tanh_gradient, 0.1)
    backprop(np.array([[1,1,1,1,1]]).T, np.array([[1, 0]]).T, weights, biases, \
             crentropy_dvt, np.tanh, tanh_dvt, softmax, softmax_gradient, 0.1)
    

if __name__ == "__main__":
    main()