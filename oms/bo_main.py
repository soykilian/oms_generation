import datetime
import os
import sys

import colorama
import matplotlib.pyplot as plt
import skopt.plots
from skopt import Optimizer
from torch.utils.data import DataLoader

from evimo_dataset import EVIMODataset
from parameter_search import create_search_space, bo_evaluate_point
from version_2.bin.activation_functions.activation_functions import *

# ---------------------
# ---- PARAMETERS -----
# ---------------------

# We are minimizing loss so set default to huge value
best_score = 10000000000000000000000000
best_config = {}
best_run_dir: str = "/home/aryalohia/PycharmProjects/iris/version_2/experiments/experiment_3_vsim/results"
num_bo_iterations: int = 500
num_initial_points: int = 100

width = 346
height = 260
fps = 40

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
print(ROOT_DIR)
print(os.getcwd())
data_dir = "/home/aryalohia/PycharmProjects/iris/version_2/train"
params_dir = os.path.join("/home/aryalohia/PycharmProjects/iris/version_2/experiments/experiment_3_vsim/data_1")
save_dir = os.path.join(os.getcwd(), 'results/')

# DATA

dataset = EVIMODataset(data_dir)
# noinspection DuplicatedCode
loader = DataLoader(dataset)

results: list = []

# Create the optimizer
optimizer = Optimizer(
    dimensions=create_search_space(),
    n_initial_points=num_initial_points
)

# Bayesian Optimization loop
for i in range(num_bo_iterations):

    # Ask the optimizer for the next point to evaluate
    next_point: list = optimizer.ask()

    # If center_radius is > surround_radius, tell Bayes Opt algorithm that its score is very bad
    if next_point[1] >= next_point[2]:
        loss = sys.maxsize
        result = optimizer.tell(next_point, sys.maxsize)
    else:
        # Evaluate the point and return a loss score
        loss = bo_evaluate_point(
            next_point,
            params_dir=params_dir,
            loader=loader,
            fps=fps,
            width=width,
            height=height,
            stride=1,
            activation_func=neuro_activation_function,
        )

        # Update the optimizer with the loss score
        result = optimizer.tell(next_point, loss)  # Returns a result object that can be used later to plot

    new_low = False

    # Save the best point
    if loss < best_score:
        new_low = True
        best_score = loss
        best_config = next_point

    # Prints the iteration number, best score so far, and best config so far
    if new_low:
        print(
            colorama.Fore.RED + "New best config: " + colorama.Fore.RESET +
            f"Iteration {i} Best Score = {best_score} Best Config = {best_config}\n"
        )
    else:
        print(f"Iteration {i} Best Score = {best_score} Best Config = {best_config}")
        print(f"Iteration {i} Loss = {loss} this_config {next_point}\n")


# Save the best config
time = datetime.datetime.now()
save_dir = os.path.join(
    best_run_dir,
    f"best_config_{time.month}-{time.day}-{time.year}_{time.hour}-{time.minute}.txt"
)

with open(save_dir, "w") as f:
    f.write("['filter_type', center_radius, surround_radius, motion surround weight, DVS threshold, "
            "OMS threshold] \n")
    f.write(f"Best Config: {best_config}\n")

# Plot convergence graph
ax = skopt.plots.plot_convergence(
    results
)

plt.show()




