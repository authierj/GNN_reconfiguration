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
    default="GatedSwitchGNN",
    choices=[
        "GCN_Global_MLP_reduced_model",
        "GCN_local_MLP",
        "GatedSwitchGNN_globalMLP",
        "GatedSwitchGNN",
        "GNN_global_MLP",
        "GNN_local_MLP",
    ],
    help="model to train",
)
parser.add_argument(
    "--warmStart", action="store_true", help="whether to warm start the PhyR"
)
parser.add_argument("--saveAllStats", action="store_true")
parser.add_argument("--saveModel", action="store_true")
parser.add_argument("--topoLoss", action="store_true")

exp_args = vars(parser.parse_args())
args = default_args()
args["lr"] = 5 * 1e-4
args["epochs"] = 1000

for key, value in exp_args.items():
    args[key] = value

print(args)
save_dir = os.path.join("results", "experiments")
if args["topoLoss"]:
    filepath = os.path.join(save_dir, "_".join(["test_topo_prob", "PhyR"]) + ".txt")
else:
    filepath = os.path.join(save_dir, "_".join(["basic", "experiments", "graphs"]) + ".txt")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

num_runs = 10

if os.path.exists(filepath):
    print("this file already exists and will be completed with new results")

for i in range(0, num_runs):
    if args["model"] in [
        "GCN_Global_MLP_reduced_model",
        "GCN_local_MLP",
        "GNN_global_MLP",
        "GNN_local_MLP",
    ]:
        result_dir = main_classical.main(args)
    else:
        result_dir = main_gated.main(args)

    data = f'lr: {args["lr"]:.0e}, hidden_features: {args["hiddenFeatures"]}, num_layers: {args["numLayers"]}, run: {i}, dir: {result_dir} \n'

    with open(filepath, "a") as f:
        f.write(data)


print("PROGRAM FINISHED")
