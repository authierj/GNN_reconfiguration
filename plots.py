from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr as pcorr
from matplotlib.ticker import FormatStrFormatter
from IPython.display import set_matplotlib_formats
import os
 

set_matplotlib_formats("retina")

list_names = [
    "GCN_Global_MLP_reduced_model.npz",
    "GCN_local_MLP.npz",
    "GatedSwitchGNN_globalMLP.npz",
    "GatedSwitchGNN.npz",
]
abreviation_names = ["GCN_global", "GCN_local", "G_GNN_global", "G_GNN_local"]
opt_gaps_name = ["pij", "qij", "v", "p_g", "q_g", "topo"]
ineq_names = ["p_g", "q_g", "p_feeder", "q_feeder", "v", "connectivity"]


training_losses = np.zeros((len(list_names), 500))
validation_losses = np.zeros((len(list_names), 500))
opt_gaps = np.zeros((len(list_names), 500, 6))
ineq_distances = np.zeros((len(list_names), 500, 6))

i = 0
for filename in list_names:
    data = np.load("trained_nn/" + filename)
    # training_losses[i, :] = data["train_losses"]
    training_losses[i, :] = data["arr_0"]
    # validation_losses[i, :] = data["valid_losses"]
    validation_losses[i, :] = data["arr_1"]
    # opt_gaps[i, :, :] = data["opt_gaps"]
    opt_gaps[i, :, :] = data["arr_2"]
    # ineq_distances[i, :, :] = data["ineq_distances"]
    ineq_distances[i, :, :] = data["arr_5"]
    i += 1

fig1, ax1 = plt.subplots(nrows=1, ncols=1)
ax1.set_yscale("log")
fig2, ax2 = plt.subplots(nrows=3, ncols=2)
fig2.set_size_inches(6 * 3, 8)
fig3, ax3 = plt.subplots(nrows=3, ncols=2)
fig3.set_size_inches(6 * 3, 8)

for i in range(len(list_names)):
    ax1.plot(training_losses[i, :], label=abreviation_names[i], color=f"C{i}")
    ax1.plot(validation_losses[i, :], "--", color=f"C{i}")
    for j in range(6):
        if i == 0:
            ax2[j // 2, j % 2].title.set_text(opt_gaps_name[j])
            ax3[j//2, j%2].title.set_text(ineq_names[j])
            # ax2[j // 2, j % 2].set_yscale("log")
            ax2[j // 2, j % 2].set_ylim([0, 50])
            ax3[j // 2, j % 2].set_yscale("log")

        ax2[j // 2, j % 2].plot(opt_gaps[i, :, j], color=f"C{i}")
        ax3[j // 2, j % 2].plot(ineq_distances[i, :, j]+1e-10, color=f"C{i}")

ax3[2,1].set_yscale("linear")
ax2[2,1].set_yscale("linear")
ax2[2,1].set_ylim([0, 0.5])

ax1.legend()
fig3.suptitle("mean inequality distances squared")
fig1.show()
fig2.show()
fig3.show()


def loss_cruve(train_losses, valid_losses, args):

    x = range(len(train_losses))

    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    fig1.set_size_inches(6, 4)

    ax1.plot(x, train_losses, label="training loss")
    ax1.plot(x, valid_losses, label="validation loss")
    ax1.legend()

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set

    plot_path = "plots/" + "_".join(
        (args["network"], args["GNN"], args["readout"], "loss_curve.png")
    )

    if os.path.isfile(plot_path):
        os.remove(plot_path)
    fig1.savefig(plot_path)


def hyperparam_plot(train_losses, valid_losses, param_name, param_values):

    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    fig1.set_size_inches(6, 4)

    for i in range(len(param_values)):
        ax1.plot(train_losses, label="training loss")
        ax1.plot(valid_losses, label="validation loss")
        ax1.legend()

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set
