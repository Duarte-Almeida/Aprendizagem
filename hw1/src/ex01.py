import numpy as np
from scipy import stats
import pandas as pd
from termcolor import colored

def read_data(filename):
    dataset = pd.read_csv(filename)
    return dataset

def estimate_parameters(dataset):

    params = []

    y1 = np.array(dataset["y1"])
    y2 = np.array(dataset["y2"])
    y3_y4 = np.array(dataset[["y3", "y4"]])
    h = np.array(dataset["class"])

    print(colored("Bayesian classifier training", "yellow", attrs = ["bold"]))
    print(colored("y1|h estimates", "green", attrs = ["bold"]))
    
    µ_y1_h0 = np.mean(y1[h == 0])
    params.append(µ_y1_h0)
    print(colored(f"µ y1|(h = 0) = {round(µ_y1_h0, 4)}", "magenta"))

    µ_y1_h1 = np.mean(y1[h == 1])
    params.append(µ_y1_h1)
    print(colored(f"µ y1|(h = 1) = {round(µ_y1_h1, 4)}", "magenta"))

    # ATTENTION: this is variance, not stddev as the notation might suggest
    σ_y1_h0 = np.sum((y1[h == 0] -  µ_y1_h0) ** 2) / (y1[h == 0].shape[0] - 1)
    params.append(σ_y1_h0)
    print(colored(f"σ y1|(h = 0) = {round(σ_y1_h0, 4)}", "magenta"))

    σ_y1_h1 = np.sum((y1[h == 1] -  µ_y1_h1) ** 2) / (y1[h == 1].shape[0] - 1)
    params.append(σ_y1_h1)
    print(colored(f"σ y1|(h = 1) = {round(σ_y1_h1, 4)}", "magenta"))

    print(colored("y3, y4|h estimates", "green", attrs = ["bold"]))
    
    µ_y3_y4_h0 = np.mean(y3_y4[h == 0])
    params.append(µ_y3_y4_h0)
    print(colored(f"µ y3, y4|(h = 0) = {round(µ_y3_y4_h0, 4)}", "magenta"))

    µ_y3_y4_h1 = np.mean(y3_y4[h == 1])
    params.append(µ_y3_y4_h1)
    print(colored(f"µ y3, y4|(h = 1) = {round(µ_y3_y4_h1, 4)}", "magenta"))

    aux_y3_y4_h0 = np.apply_along_axis(lambda row: np.atleast_2d(row - µ_y3_y4_h0).T.dot(np.atleast_2d(row - µ_y3_y4_h0)), 1, y3_y4)
    Σ_y3_y4_h0 = np.apply_along_axis(np.sum, 0, aux_y3_y4_h0) 
    params.append(params.append(Σ_y3_y4_h0))
    print(colored(f"Σ y3, y4|(h = 0) = {round(σ_y1_h0, 4)}", "magenta"))

    aux_y3_y4_h1 = np.apply_along_axis(lambda row: np.atleast_2d(row - µ_y3_y4_h1).T.dot(np.atleast_2d(row - µ_y3_y4_h1)), 1, y3_y4)
    Σ_y3_y4_h1 = np.apply_along_axis(np.sum, 0, aux_y3_y4_h1) 
    params.append(params.append(Σ_y3_y4_h1))
    print(colored(f"Σ y3, y4|(h = 1) = {round(σ_y1_h1, 4)}", "magenta"))

    print(colored("P(y2 | h) estimates", "green", attrs = ["bold"]))
    p_y2_A_h0 = np.mean(np.vectorize(lambda x : 1 if x == "A" else 0)(y2[h == 0]))
    p_y2_B_h0 = np.mean(np.vectorize(lambda x : 1 if x == "B" else 0)(y2[h == 0]))
    p_y2_C_h0 = np.mean(np.vectorize(lambda x : 1 if x == "C" else 0)(y2[h == 0]))

    p_y2_A_h1 = np.mean(np.vectorize(lambda x : 1 if x == "A" else 0)(y2[h == 1]))
    p_y2_B_h1 = np.mean(np.vectorize(lambda x : 1 if x == "B" else 0)(y2[h == 1]))
    p_y2_C_h1 = np.mean(np.vectorize(lambda x : 1 if x == "C" else 0)(y2[h == 1]))

    params.append(p_y2_A_h0)
    params.append(p_y2_B_h0)
    params.append(p_y2_C_h0)
    params.append(p_y2_A_h1)
    params.append(p_y2_B_h1)
    params.append(p_y2_C_h1)

    print(colored(f"P(y2 = A | h = 0) = {p_y2_A_h0}", "magenta"))
    print(colored(f"P(y2 = B | h = 0) = {p_y2_B_h0}", "magenta"))
    print(colored(f"P(y2 = C | h = 0) = {p_y2_C_h0}", "magenta"))
    print(colored(f"P(y2 = A | h = 1) = {p_y2_A_h1}", "magenta"))
    print(colored(f"P(y2 = B | h = 1) = {p_y2_B_h1}", "magenta"))
    print(colored(f"P(y2 = C | h = 1) = {p_y2_C_h1}", "magenta"))

    print(colored("h estimates", "green", attrs = ["bold"]))
    print(colored(f"P(h = 1) = {np.mean(h)} \nP(h = 0) = {1 - np.mean(h)}", "magenta"))

    params.append(np.mean(h))
    params.append(1 - np.mean(h))

    return params

def main():
    dataset = read_data("../data/data.csv")
    print(dataset)
    params = estimate_parameters(dataset)

if __name__ == "__main__":
    main()