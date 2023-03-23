import numpy as np
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats


from utils_JA import default_args
from main_edge_features import main

set_matplotlib_formats("retina")

args = default_args()

args["epochs"] = 1000
args["hiddenFeatures"] = 16
args["numLayers"] = 10
args["aggregation"] = "max"
# test_param = "lr"
# lr = [1e-2, 1e-3]

# train_losses = np.zeros((len(lr), args["epochs"]))
# valid_losses = np.zeros((len(lr), args["epochs"]))

# fig1, ax1 = plt.subplots(nrows=1, ncols=1)
# fig1.set_size_inches(6, 4)


# for i in range(len(lr)):
#     args[test_param] = lr[i]
#     train_losses[i, :], valid_losses[i, :] = main(args)
#     ax1.plot(train_losses[i, :], color=f"C{i}", label=test_param + f"_{lr[i]}")
#     ax1.plot(train_losses[i, :], color=f"C{i}", linestyle='--')


# title = "_".join((args["network"], args["GNN"], args["readout"], "loss_curve.png"))
# ax1.set_title(title)
# ax1.legend()

train_losses, valid_losses, opt_gaps, line_losses, train_losses_ineq, ineq_distances = main(args)

fig1, ax1 = plt.subplots(nrows=1, ncols=1)
fig1.set_size_inches(6, 4)
ax1.plot(train_losses)
ax1.plot(train_losses, linestyle='--')
# ax1.plot(line_losses, linestyle='--')
# ax1.plot(train_losses_ineq, linestyle='-.')
# ax1.set_yscale('log')
fig1.show()


fig2, ax2 = plt.subplots(nrows=1, ncols=1)
fig2.set_size_inches(6, 4)
ax2.plot(ineq_distances[:,0], label="pg")
ax2.plot(ineq_distances[:,1], label="qg")
ax2.plot(ineq_distances[:,2], label="p_feeder")
ax2.plot(ineq_distances[:,3], label="q_feeder")
ax2.plot(ineq_distances[:,4], label="v")
ax2.plot(ineq_distances[:,5], label="connnect")
# ax2.set_yscale('log')
ax2.legend()
fig2.show()

# fig3, ax3 = plt.subplots(nrows=1, ncols=1)
# fig3.set_size_inches(6, 4)
# ax3.plot(ineq_distances[:,1])
# # ax3.set_yscale('log')
# fig3.show()

# fig4, ax4 = plt.subplots(nrows=1, ncols=1)
# fig4.set_size_inches(6, 4)
# ax4.plot(ineq_distances[:,2])
# # ax4.set_yscale('log')
# fig4.show()

# fig5, ax5 = plt.subplots(nrows=1, ncols=1)
# fig5.set_size_inches(6, 4)
# ax5.plot(ineq_distances[:,3])
# # ax5.set_yscale('log')
# fig5.show()

# fig6, ax6 = plt.subplots(nrows=1, ncols=1)
# fig6.set_size_inches(6, 4)
# ax6.plot(ineq_distances[:,4])
# # ax6.set_yscale('log')
# fig6.show()


# fig7, ax7 = plt.subplots(nrows=1, ncols=1)
# fig7.set_size_inches(6, 4)
# ax7.plot(ineq_distances[:,5])
# # ax7.set_yscale('log')
# fig7.show()
