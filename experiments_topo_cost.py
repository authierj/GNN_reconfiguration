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
        "--switchActivation", type=str, default=None, choices=["sig", "mod_sig","None"]
)

batch_args = vars(parser.parse_args())
args = default_args()

try:
    assert args["topoLoss"] 
except AssertionError:
    print("this file is to test the topoLoss!")
    exit()

save_dir = os.path.join("results", "experiments")
filepath = os.path.join(save_dir, "_".join(["supervised",f'{args["switchActivation"]}', "mod_PhyR"])+".txt")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

num_runs = 5
args["model"] = batch_args["model"]
args["switchActivation"] = batch_args["switchActivation"]

if os.path.exists(filepath):
    print("this file already exists and will be completed with new results")

for i in range(0, num_runs):
    if args["model"] in ["GCN_Global_MLP_reduced_model", "GCN_local_MLP"]:
        result_dir = main_classical.main(args)
    else:
        result_dir = main_gated.main(args)

    data = f'lr: {args["lr"]:.0e}, hidden_features: {args["hiddenFeatures"]}, num_layers: {args["numLayers"]}, run: {i}, dir: {result_dir} \n'

    with open(filepath, "a") as f:
        f.write(data)


print("PROGRAM FINISHED")
