from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr as pcorr
from matplotlib.ticker import FormatStrFormatter
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('retina')


def loss_cruve(train_losses, valid_losses, args):

    x = range(len(train_losses))

    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    fig1.set_size_inches(6, 4)

    ax1.plot(x, train_losses, label='training loss')
    ax1.plot(x, valid_losses, label='validation loss')
    ax1.legend()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set

    fig1.savefig('plots/' + args['GNN'] + '_' + args['readout'] + '_' + 'loss_curve.png' )
