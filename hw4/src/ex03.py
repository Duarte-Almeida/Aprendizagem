import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True

dim_1 = [2,5,10,12,13]
dim_2 = [2,5,10,30, 100,300,1000]

def vc_dimension_MLP(x):
    return np.vectorize(lambda el: 3 * el ** 2 + 5 * el + 2)(x)

def vc_dimension_DT(x):
    return np.vectorize(lambda el: 3 ** el)(x)

def vc_dimension_Bayes(x):
    return np.vectorize(lambda el: el ** 2 + 3 * el + 1)(x)

def main():

    fig, ax = plt.subplots()
    ax.plot(dim_1, vc_dimension_Bayes(dim_1), label =  "$d_{\\mathrm{VC}}(\\mathrm{Bayesian \; Classifier})$", linewidth=2)
    ax.plot(dim_1, vc_dimension_MLP(dim_1), label = "$d_{\\mathrm{VC}}(\\mathrm{MLP})$", linestyle = "dashed", linewidth=1, markersize=12)
    ax.plot(dim_1, vc_dimension_DT(dim_1), label =  "$d_{\\mathrm{VC}}(\\mathrm{Decision \; Tree})$", linewidth=2)
    ax.set_xlabel("$m\;(\mathrm{data\;dimensionality})$")
    ax.set_ylabel("$d_{\mathrm{VC}}\;(\mathrm{VC\;dimension})$")
    ax.legend(loc = "best")
    plt.savefig("output/vc_4_a.pdf")

    plt.cla()

    plt.plot(dim_2, vc_dimension_MLP(dim_2), label = "$d_{\\mathrm{VC}}(\\mathrm{MLP})$")
    plt.plot(dim_2, vc_dimension_Bayes(dim_2), label =  "$d_{\\mathrm{VC}}(\\mathrm{Bayesian \; Classifier})$")
    ax.set_xlabel("$m\;(\mathrm{data\;dimensionality})$")
    ax.set_ylabel("$d_{\mathrm{VC}}\;(\mathrm{VC\;dimension})$")

    plt.legend(loc = "best")
    plt.savefig("output/vc_4_b.pdf")
    

if __name__ == "__main__":
    main()