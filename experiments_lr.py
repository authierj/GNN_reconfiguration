# file to set up experiments and run the main method
import main_reduced_model
import os
from utils_JA import default_args
import numpy as np

exp_name = "GCN_local_MLP_hidden_features_test"
save_dir = os.path.join("results", "experiments")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

filepath = os.path.join(save_dir, exp_name + ".txt")
if os.path.exists(filepath):
    print("Error: experiment name/version already exists")
    exit(1)

# exponents = np.arange(-6, 0)
# lr_params = 10.0**exponents
num_features = np.arange(2,14,2)
num_runs = 10
num_params = num_features.shape[0]
args = default_args()

for j in range(0, num_params):
    args["hiddenFeatures"] = num_features[j]
    for i in range(0, num_runs):
        result_dir = main_reduced_model.main(args)
        data = f'hidden_features: {args["hiddenFeatures"]} num_layers: {args["numLayers"]}, run: {i+1}, dir: {result_dir} \n'

        with open(filepath, "a") as f:
            f.write(data)


print("PROGRAM FINISHED")