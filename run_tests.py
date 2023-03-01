import numpy as np
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats


from utils_JA import default_args
from main import main

set_matplotlib_formats("retina")

args = default_args()

args["epochs"] = 100
test_param = "lr"
lr = [1e-2, 1e-3]

train_losses = np.zeros((len(lr), args["epochs"]))
valid_losses = np.zeros((len(lr), args["epochs"]))

fig1, ax1 = plt.subplots(nrows=1, ncols=1)
fig1.set_size_inches(6, 4)


for i in range(len(lr)):
    args[test_param] = lr[i]
    train_losses[i, :], valid_losses[i, :] = main(args)
    ax1.plot(train_losses[i, :], color=f"C{i}", label=test_param + f"_{lr[i]}")
    ax1.plot(train_losses[i, :], color=f"C{i}", linestyle='--')


title = "_".join((args["network"], args["GNN"], args["readout"], "loss_curve.png"))
ax1.set_title(title)
ax1.legend()
