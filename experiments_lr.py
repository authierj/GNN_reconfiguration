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
# end for: runs


# exp_name = 'exp_BW33_PhMLDyR_supervised_2HL_NNsize_ver2'  ## redo with a penalty for inequality constraint violation
# save_dir = os.path.join('results', 'experiments')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# filepath = os.path.join(save_dir, exp_name + '.txt')
# if os.path.exists(filepath):
#     print('Error: experiment name/version already exists')
#     exit(1)
#
# NN_size = 5
# num_runs = 10
# num_epochs = 2500
# lr_param = 1e-4
# for j in range(0, num_runs):
#     result_dir = method.main(network="baranwu33", hiddenSize=NN_size, epochs=num_epochs,
#                              dataType="perturbed", lr=lr_param, lossFnc='supervised')
#     data = 'hidden layer size: ' + str(NN_size) + ', run: ' + str(j+1) + ', dir: ' + result_dir + '\n'
#
#     with open(filepath, 'a') as f:
#         f.write(data)
#     # end for: runs
# # end for: params


# exp_name = 'exp_BW33_SigFnc_supervised_2HL_NNsize_ver2'
# save_dir = os.path.join('results', 'experiments')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# filepath = os.path.join(save_dir, exp_name + '.txt')
# if os.path.exists(filepath):
#     print('Error: experiment name/version already exists')
#     exit(1)
#
# NN_size = [5, 25]
# num_runs = 10
# num_epochs = 3500
# lr_param = 1e-4
# for NN_s in NN_size:
#     print('starting NN size class:{}'.format(NN_s))
#     for j in range(0, num_runs):
#         result_dir = method.main(network="baranwu33", hiddenSize=NN_s, epochs=num_epochs,
#                                  dataType="perturbed", usePhML=False, lr=lr_param, lossFnc='supervised')
#         data = 'hidden layer size: ' + str(NN_s) + ', run: ' + str(j+1) + ', dir: ' + result_dir + '\n'
#
#         with open(filepath, 'a') as f:
#             f.write(data)
#     # end for: runs
# # end for: params


## BaranWu33
# exp_name = 'exp_BW33_SigFnc_unsupervised_2HL_NNsize_ver2'
# # exp_name = 'exp_BW33_PhMLDyR_unsupervised_NNsize_ver3'
# # exp_name = 'exp_BW33_PhMLDyR_unsupervised_2HL_NNsize_ver2'
# save_dir = os.path.join('results', 'experiments')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# filepath = os.path.join(save_dir, exp_name + '.txt')
# if os.path.exists(filepath):
#     print('Error: experiment name/version already exists')
#     exit(1)
#
# NN_size = [5, 25, 100, 300]
# num_runs = 10
# num_epochs = 1500
# lr_param = 1e-4
# for NN_s in NN_size:
#     print('starting NN size class:{}'.format(NN_s))
#     for j in range(0, num_runs):
#         result_dir = method.main(network="baranwu33", hiddenSize=NN_s, epochs=num_epochs, dataType="perturbed",
#                                  usePhML=False, lr=lr_param)
#         data = 'hidden layer size: ' + str(NN_s) + ', run: ' + str(j+1) + ', dir: ' + result_dir + '\n'
#
#         with open(filepath, 'a') as f:
#             f.write(data)
#     # end for: runs
# # end for: params


#
# # adding a training set for 83 node to parallelize
# exp_name = 'exp_REDS83_PhMLDyR_unsupervised_2HL_PGcapacity_ver2'
# save_dir = os.path.join('results', 'experiments')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# filepath = os.path.join(save_dir, exp_name + '.txt')
# if os.path.exists(filepath):
#     print('Error: experiment name/version already exists')
#     exit(1)
#
# NN_size = 300
# num_runs = 10
# num_epochs = 1500
# for j in range(0, num_runs):
#     result_dir = method.main(network="node83REDS", hiddenSize=NN_size, epochs=num_epochs, dataType="profile",
#                              adjustDataSize=True, trainDataSize=9000, usePhML=True, lr=1e-4)
#     data = 'hidden layer size: ' + str(NN_size) + ', run: ' + str(j+1) + ', dir: ' + result_dir + '\n'
#
#     with open(filepath, 'a') as f:
#         f.write(data)
# # end for: runs

print("PROGRAM FINISHED")