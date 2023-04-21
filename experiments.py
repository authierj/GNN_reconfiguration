# file to set up experiments and run the main method
import main_classical
import main_gated
import os
from utils_JA import default_args
import numpy as np
import argparse

# create the parser
parser = argparse.ArgumentParser()

# add the arguments to the parser
parser.add_argument(
    "--model",
    type=str,
    default="GCN_local_MLP",
    choices=[
        "GCN_Global_MLP_reduced_model",
        "GCN_local_MLP",
        "GatedSwitchGNN_globalMLP",
        "GatedSwitchGNN",
    ],
    help="model to train",
)
parser.add_argument(
    "--param",
    type=str,
    default="lr",
    choices=["lr", "hiddenFeatures", "numLayers"],
    help="parameter to test",
)

batch_args = vars(parser.parse_args())
param = batch_args["param"]
args = default_args()


if param == "lr":
    exp_name = "_".join((batch_args["model"], param, "test"))
    exponents = np.arange(-6, 0)
    param_list = 10.0**exponents
elif param == "hiddenFeatures":
    exp_name = "_".join((batch_args["model"], param, "test"))
    exponents = np.arange(1, 7).astype(int)
    print(exponents[0].dtype)
    param_list = 2.0**exponents
elif param == "numLayers":
    exp_name = "_".join((batch_args["model"], param, "test"))
    param_list = np.arange(2, 14, 2)
else:
    print("parameter not recognized")
    exit()

try:
    assert param_list.shape[0] == 6
except AssertionError:
    print(param + "has wrong number of parameters")
    exit()

save_dir = os.path.join("results", "experiments")
filepath = os.path.join(save_dir, exp_name + ".txt")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

num_runs = 10
args["model"] = batch_args["model"]

if os.path.exists(filepath):
    print("this file already exists and will be completed with new results")
for j in range(param_list.shape[0]):
    if param == "lr":
        args[param] = param_list[j]
    else:
        args[param] = int(param_list[j])
    for i in range(0, num_runs):
        if args["model"] in ["GCN_Global_MLP_reduced_model", "GCN_local_MLP"]:
            result_dir = main_classical.main(args)
        else:
            result_dir = main_gated.main(args)

        data = f'lr: {args["lr"]:.0e}, hidden_features: {args["hiddenFeatures"]}, num_layers: {args["numLayers"]}, run: {i}, dir: {result_dir} \n'

        with open(filepath, "a") as f:
            f.write(data)


print("PROGRAM FINISHED")
