import numpy as np
from scipy import stats
import pandas as pd
from termcolor import colored

# Bayesian classifier parameters
µ_y1 = {0: None, 1: None} 
σ_y1 = {0: None, 1: None} 

µ_y3_y4 = {0: None, 1: None} 
Σ_y3_y4 = {0: None, 1: None} 

p_y2 = {0: {"A": None, "B": None, "C": None, "D": None}, 1: {"A": None, "B": None, "C": None, "D": None}}

p_h = {0: None, 1: None} 

def read_data(filename):
    dataset = pd.read_csv(filename)
    return dataset

def estimate_parameters(dataset):

    y1 = np.array(dataset["y1"])
    y2 = np.array(dataset["y2"])
    y3_y4 = np.array(dataset[["y3", "y4"]])
    h = np.array(dataset["class"])

    print(colored("Bayesian classifier training", "yellow", attrs = ["bold"]))
    print(colored("y1|h estimates", "green", attrs = ["bold"]))
    
    µ_y1[0] = np.mean(y1[h == 0])
    print(colored(f"µ y1|(h = 0) = {round(µ_y1[0], 4)}", "magenta"))

    µ_y1[1] = np.mean(y1[h == 1])
    print(colored(f"µ y1|(h = 1) = {round(µ_y1[1], 4)}", "magenta"))

    # ATTENTION: this is variance, not stddev as the notation might suggest
    σ_y1[0] = np.sum((y1[h == 0] -  µ_y1[0]) ** 2) / (y1[h == 0].shape[0] - 1)
    print(colored(f"σ y1|(h = 0) = {round(σ_y1[0], 4)}", "magenta"))

    σ_y1[1] = np.sum((y1[h == 1] -[1]) ** 2) / (y1[h == 1].shape[0] - 1)
    print(colored(f"σ y1|(h = 1) = {round(σ_y1[1], 4)}", "magenta"))

    print(colored("y3, y4|h estimates", "green", attrs = ["bold"]))
    
    µ_y3_y4[0] = np.sum(y3_y4[h == 0], axis = 0) / y3_y4[h == 0].shape[0]
    print(colored(f"µ y3, y4|(h = 0) = {µ_y3_y4[0]}", "magenta"))

    µ_y3_y4[1] = np.sum(y3_y4[h == 1], axis = 0) / y3_y4[h == 1].shape[0]
    print(colored(f"µ y3, y4|(h = 1) = {µ_y3_y4[1]}", "magenta"))

    aux_y3_y4_h0 = np.apply_along_axis(lambda row: np.atleast_2d(row - µ_y3_y4[0]).T.dot(np.atleast_2d(row - µ_y3_y4[0])), 1, y3_y4)
    Σ_y3_y4[0] = np.apply_along_axis(np.sum, 0, aux_y3_y4_h0) 
    print(colored(f"Σ y3, y4|(h = 0) = {round(σ_y1[0], 4)}", "magenta"))

    aux_y3_y4_h1 = np.apply_along_axis(lambda row: np.atleast_2d(row - µ_y3_y4[1]).T.dot(np.atleast_2d(row - µ_y3_y4[1])), 1, y3_y4)
    Σ_y3_y4[1] = np.apply_along_axis(np.sum, 0, aux_y3_y4_h1) 
    print(colored(f"Σ y3, y4|(h = 1) = {round(σ_y1[1], 4)}", "magenta"))

    print(colored("P(y2 | h) estimates", "green", attrs = ["bold"]))
    p_y2[0]["A"] = np.mean(np.vectorize(lambda x : 1 if x == "A" else 0)(y2[h == 0]))
    p_y2[0]["B"] = np.mean(np.vectorize(lambda x : 1 if x == "B" else 0)(y2[h == 0]))
    p_y2[0]["C"] = np.mean(np.vectorize(lambda x : 1 if x == "C" else 0)(y2[h == 0]))

    p_y2[1]["A"] = np.mean(np.vectorize(lambda x : 1 if x == "A" else 0)(y2[h == 1]))
    p_y2[1]["B"] = np.mean(np.vectorize(lambda x : 1 if x == "B" else 0)(y2[h == 1]))
    p_y2[1]["C"] = np.mean(np.vectorize(lambda x : 1 if x == "C" else 0)(y2[h == 1]))

    print(colored("P(y2 = A | h = 0) = {}".format(p_y2[0]["A"]), "magenta"))
    print(colored("P(y2 = B | h = 0) = {}".format(p_y2[0]["B"]), "magenta"))
    print(colored("P(y2 = C | h = 0) = {}".format(p_y2[0]["C"]), "magenta"))
    print(colored("P(y2 = A | h = 1) = {}".format(p_y2[1]["A"]), "magenta"))
    print(colored("P(y2 = B | h = 1) = {}".format(p_y2[1]["B"]), "magenta"))
    print(colored("P(y2 = C | h = 1) = {}".format(p_y2[1]["C"]), "magenta"))

    p_h[0] = 1 - np.mean(h)
    p_h[1] = np.mean(h)

    print(colored("h estimates", "green", attrs = ["bold"]))
    print(colored(f"P(h = 1) = {np.mean(h)} \nP(h = 0) = {1 - np.mean(h)}", "magenta"))

def estimate_instances(dataset):

    print(colored("Classifying instances", "yellow", attrs = ["bold"]))
    inputs = np.array(dataset[["y1", "y2", "y3", "y4"]])
    outputs = np.array(dataset["class"])

    posterior_h0 = np.apply_along_axis(lambda instance: bayesian_classification(instance, 0), 1, inputs)
    posterior_h1 = np.apply_along_axis(lambda instance: bayesian_classification(instance, 1), 1, inputs)

    estimates = np.vectorize(lambda x: 0 if x == True else 1)(posterior_h0 > posterior_h1)
    estimates_df = pd.DataFrame(np.array([outputs, estimates]).T, index = [f"x{i}" for i in range(1, outputs.shape[0] + 1)], columns = ["Class", "Estimate"])
    print(estimates_df)
    estimates_df.to_csv("output/estimates.csv")


def bayesian_classification(instance, hipothesys):
    return stats.norm.pdf(instance[0], loc = µ_y1[hipothesys], scale = np.sqrt(σ_y1[hipothesys])) * \
           p_y2[hipothesys][instance[1]] * \
           stats.multivariate_normal(mean = µ_y3_y4[hipothesys], cov = Σ_y3_y4[hipothesys]).pdf(np.array([instance[2], instance[3]]))

def main():
    dataset = read_data("../data/data.csv")
    print(colored("Data", "yellow", attrs = ["bold"]))
    print(dataset)
    estimate_parameters(dataset)
    estimate_instances(dataset)

if __name__ == "__main__":
    main()