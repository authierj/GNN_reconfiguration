# file to set up experiments and run the main method
import main_gated
import os
from utils_JA import default_args
import numpy as np

save_dir = os.path.join("results", "experiments")
models = ["GatedSwitchGNN_globalMLP", "GatedSwitchGNN"]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

features = np.arange(2, 12, 2)
num_runs = 10
num_params = features.shape[0]
args = default_args()

for model in models:
    exp_name = "_".join((model, "features_test"))
    args["model"] = model
    filepath = os.path.join(save_dir, exp_name + ".txt")
    if os.path.exists(filepath):
        print("Error: experiment name/version already exists")
        exit(1)
    for j in range(num_params):
        args["hiddenFeatures"] = features[j]
        for i in range(0, num_runs):
            result_dir = main_gated.main(args)
            data = f'hidden_features: {args["hiddenFeatures"]} num_layers: {args["numLayers"]}, run: {i+1}, dir: {result_dir} \n'

            with open(filepath, "a") as f:
                f.write(data)

print("PROGRAM FINISHED")
