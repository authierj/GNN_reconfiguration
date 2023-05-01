# file to parse results from experiments and produce plots
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
from utils_JA import dict_agg


def main():
    # exp_names = ["GatedSwitchGNN_globalMLP_lr_test","GatedSwitchGNN_globalMLP_numLayers_test", "GatedSwitchGNN_globalMLP_hiddenFeatures_test"]
    # exp_names = ["GatedSwitchGNN_lr_test","GatedSwitchGNN_numLayers_test", "GatedSwitchGNN_hiddenFeatures_test"]
    # exp_names = ["GCN_Global_MLP_reduced_model_numLayers_test", "GCN_Global_MLP_reduced_model_hiddenFeatures_test"]
    exp_names = ["supervised_sig_mod_PhyR"]
    save_dir = "results/experiments"
    filepaths = [os.path.join(save_dir, e_name + ".txt") for e_name in exp_names]
    f_exist = [f for f in filepaths if os.path.isfile(f)]
    f_not_exist = list(set(f_exist) ^ set(filepaths))

    if len(f_not_exist) > 0:
        print("Error: experiment name/version not found:")
        print(f_not_exist)
        exit(1)
    else:
        # parse_pareto_datasize_NN(filepaths)
        for fp in filepaths:
            print(fp)
            parse_NN_size([fp])
        # parse_NN_size(filepaths)  # currently only for a single file at a time
        # parse_pareto_PhMethod(filepaths)
    return


def parse_NN_size(experiment_filepath):
    if len(experiment_filepath) > 1:
        print("Warning: only first experiment file will be plotted")

    exp_stats = {}  # save the stats per run for an experiment
    # read file after line ###
    flag_start = False
    exp_counter = 0
    run_counter = 0
    best_topo = np.zeros(500)
    worst_topo = np.zeros(500)
    current_nn = ""
    with open(experiment_filepath[0]) as exp_file:
        while line := exp_file.readline().rstrip():
            if not flag_start:
                if line == "###":
                    flag_start = True
            elif line[0] == "#":
                continue
            else:
                # print(line)
                exp_dir = line[line.find("dir: ") + 13 :]
                exp_filepath = os.path.join("results", exp_dir, "stats.dict")
                exp_filepath_small = os.path.join(exp_dir, " .dict")
                if not os.path.exists(exp_filepath) and not os.path.exists(
                    exp_filepath_small
                ):
                    print("Error: experiment line {} not found".format(exp_counter))
                else:
                    exp_filepath_get = (
                        exp_filepath
                        if os.path.exists(exp_filepath)
                        else exp_filepath_small
                    )
                    with open(exp_filepath_get, "rb") as exp_handle:
                        exp_nn = line[line.find("supervised") : line.find("GCN")-1]
                        # exp_nn = line[line.find("lr: ") : line.find(", run")]
                        # exp_nn = line[line.find("dir: ") + 13 : -3]
                        exp_run = line[line.find("run: ") + 5 : line.find(", dir")]

                        if exp_nn != current_nn and current_nn != "":
                            # plot previous experiment results
                            ## Plotting with new stats (i.e. save results per epoch only)
                            # fmt: off
                            exp_stats["T_topology_best"] = best_topo 
                            exp_stats["T_topology_worst"] = worst_topo
                            exp_stats["T_loss_var"] = np.var(exp_stats["T_loss_var"], axis=0)
                            exp_stats["V_loss_var"] = np.var(exp_stats["V_loss_var"], axis=0)
                            exp_stats["T_dispatch_mean_var"] = np.var(exp_stats["T_dispatch_mean_var"], axis=0)
                            exp_stats["V_dispatch_mean_var"] = np.var(exp_stats["V_dispatch_mean_var"], axis=0)
                            exp_stats["T_topology_mean_var"] = np.var(exp_stats["T_topology_mean_var"], axis=0)
                            exp_stats["V_topology_mean_var"] = np.var(exp_stats["V_topology_mean_var"], axis=0)
                            # fmt: on
                            plot_exp_NNsize(
                                exp_stats, run_counter, current_nn, exp_counter
                            )
                            exp_counter += 1

                            # reset for new experiment
                            current_nn = exp_nn  # update experiment
                            exp_stats = {}  # reset the experiment stats
                            run_counter = 0  # reset run counter
                            best_topo = 0
                            worst_topo = 0

                        if current_nn == "":
                            current_nn = exp_nn

                        # save the stats per run for the experiment
                        # stats_dict = np.load(exp_handle, allow_pickle=True)
                        stats_dict = pickle.load(exp_handle)  # load the stats
                        # fmt: off
                        dict_agg(exp_stats, 'T_loss', stats_dict["train_loss"], op="sum")
                        dict_agg(exp_stats, 'V_loss', stats_dict["valid_loss"], op="sum")
                        dict_agg(exp_stats, 'T_dispatch_mean', stats_dict["train_dispatch_error_mean"], op="sum")
                        dict_agg(exp_stats, 'V_dispatch_mean', stats_dict["valid_dispatch_error_mean"], op="sum")
                        dict_agg(exp_stats, 'T_topology_mean', stats_dict["train_topology_error_mean"], op="sum")
                        dict_agg(exp_stats, 'T_topology_max', stats_dict["train_topology_error_max"], op="sum")
                        dict_agg(exp_stats, 'T_topology_min', stats_dict["train_topology_error_min"], op="sum")
                        dict_agg(exp_stats, 'V_topology_mean', stats_dict["valid_topology_error_mean"], op="sum")
                        dict_agg(exp_stats, 'V_ineq_num_viol_0', stats_dict["valid_ineq_num_viol_0"], op="sum")
                        dict_agg(exp_stats, 'V_ineq_num_viol_1', stats_dict["valid_ineq_num_viol_1"], op="sum")
                        dict_agg(exp_stats, 'V_ineq_mag_max', stats_dict["valid_ineq_max"], op="sum")
                        dict_agg(exp_stats, 'V_ineq_mag_mean', stats_dict["valid_ineq_mean"], op="sum")
                        dict_agg(exp_stats, 'T_loss_var', stats_dict["train_loss"].copy(), op="vstack")
                        dict_agg(exp_stats, 'V_loss_var', stats_dict["valid_loss"].copy(), op="vstack")
                        dict_agg(exp_stats, 'T_dispatch_mean_var', stats_dict["train_dispatch_error_mean"].copy(), op="vstack")
                        dict_agg(exp_stats, 'V_dispatch_mean_var', stats_dict["valid_dispatch_error_mean"].copy(), op="vstack")
                        dict_agg(exp_stats, 'T_topology_mean_var', stats_dict["train_topology_error_mean"].copy(), op="vstack")
                        dict_agg(exp_stats, 'V_topology_mean_var', stats_dict["valid_topology_error_mean"].copy(), op="vstack")
                        # fmt: on
                        if run_counter == 0:
                            best_topo = stats_dict["train_topology_error_mean"].copy()
                            worst_topo = stats_dict["train_topology_error_mean"].copy()
                        elif stats_dict["train_topology_error_mean"][-1] < best_topo[-1]:
                            best_topo = stats_dict["train_topology_error_mean"].copy()
                        elif stats_dict["train_topology_error_mean"][-1] > worst_topo[-1]:
                            worst_topo = stats_dict["train_topology_error_mean"].copy()
                        run_counter += 1  # increment run counter


        if flag_start:
            exp_stats["T_topology_best"] = best_topo 
            exp_stats["T_topology_worst"] = worst_topo
            exp_stats["T_loss_var"] = np.var(exp_stats["T_loss_var"], axis=0)
            exp_stats["V_loss_var"] = np.var(exp_stats["V_loss_var"], axis=0)
            exp_stats["T_dispatch_mean_var"] = np.var(
                exp_stats["T_dispatch_mean_var"], axis=0
            )
            exp_stats["V_dispatch_mean_var"] = np.var(
                exp_stats["V_dispatch_mean_var"], axis=0
            )
            exp_stats["T_topology_mean_var"] = np.var(
                exp_stats["T_topology_mean_var"], axis=0
            )
            exp_stats["V_topology_mean_var"] = np.var(
                exp_stats["V_topology_mean_var"], axis=0
            )
            plot_exp_NNsize(exp_stats, run_counter, current_nn, exp_counter)

    if flag_start:
        begin = exp_filepath.find("/") + 1
        end = exp_filepath[begin::].find("/")
        gnn = exp_filepath[begin : begin + end]
        # add legend and title to the plots
        plt.figure(1)  # train loss
        plt.legend(loc="upper right")
        plt.title(gnn + ", Training loss, mean across samples")
        plt.figure(2)  # valid loss
        plt.legend(loc="upper right")
        plt.title(gnn + ", Validation loss, mean across samples")
        plt.figure(3)  # T & V dispatch error, log scale
        plt.legend(loc="upper right")
        plt.title(gnn + ", Dispatch error, log-y")
        plt.figure(4)  # T & V topology error
        plt.legend(loc="upper right")
        plt.title(gnn + ", Topology error")
        plt.figure(5)  # V ineq error violation
        plt.legend(loc="upper right")
        plt.title(gnn + ", Inequality error violation, 0.001")
        plt.figure(6)  # V ineq error violation
        plt.legend(loc="upper right")
        plt.title(gnn + ", Inequality error violation, 0.01")

        plt.show()  # enter non-interactive mode, and keep plots
    else:
        print("ERROR: File start ### not found")

    return


def plot_exp_NNsize(exp_stats, run_counter, current_nn, exp_counter):
    x = np.arange(exp_stats["T_loss"].shape[0])
    plt.figure(1)  # train loss
    plt.yscale("log")
    plt.plot(
        exp_stats["T_loss"] / run_counter,
        label=current_nn,
        color=f"C{exp_counter}",
    )
    plt.fill_between(
        x=x,
        y1=-np.sqrt(exp_stats["T_loss_var"]) + exp_stats["T_loss"] / run_counter,
        y2=np.sqrt(exp_stats["T_loss_var"]) + exp_stats["T_loss"] / run_counter,
        color=f"C{exp_counter}",
        alpha=0.2,
    )
    plt.figure(2)  # valid loss
    plt.yscale("log")
    plt.plot(
        exp_stats["V_loss"] / run_counter,
        color=f"C{exp_counter}",
        label=current_nn,
    )
    plt.ylim([5*1e0,1e4])
    plt.fill_between(
        x=x,
        y1=-np.sqrt(exp_stats["V_loss_var"]) + exp_stats["V_loss"] / run_counter,
        y2=np.sqrt(exp_stats["V_loss_var"]) + exp_stats["V_loss"] / run_counter,
        color=f"C{exp_counter}",
        alpha=0.2,
    )
    plt.figure(3)  # T & V dispatch error, log scale
    plt.yscale("log")
    plt.plot(
        exp_stats["T_dispatch_mean"] / run_counter,
        color=f"C{exp_counter}",
    )
    plt.plot(
        exp_stats["V_dispatch_mean"] / run_counter, "--", color=f"C{exp_counter}"
    )  #  + '-V'
    plt.figure(4)  # T & V topology error
    plt.plot(
        exp_stats["T_topology_mean"] / run_counter,
        color=f"C{exp_counter}",
    )
    # plt.plot(
    #     exp_stats["V_topology_mean"] / run_counter, "--", color=f"C{exp_counter}"
    # )  # + '-V'
    # plt.fill_between(
    #     x=x,
    #     y1=-np.sqrt(exp_stats["T_topology_mean_var"])
    #     + exp_stats["T_topology_mean"] / run_counter,
    #     y2=np.sqrt(exp_stats["T_topology_mean_var"])
    #     + exp_stats["T_topology_mean"] / run_counter,
    #     color=f"C{exp_counter}",
    #     alpha=0.2,
    # )
    plt.plot(exp_stats["T_topology_best"],'-.', color=f"C{exp_counter}")
    plt.plot(exp_stats["T_topology_worst"],'--', color=f"C{exp_counter}")


    plt.figure(5)  # V ineq error violation
    plt.plot(exp_stats["V_ineq_num_viol_0"]/ run_counter)
    plt.figure(6)  # V ineq error violation
    plt.plot(exp_stats["V_ineq_num_viol_1"] / run_counter)

    # print final epoch results:
    print("\n\n ---------- \nNN size: {}".format(current_nn))
    print(
        "\n total loss = {} \n dispatch error = {} \n topology error = {} \n ineq viol"
        "mean = {} \n ineq viol max = {} \n ineq viol 0.01 = {}".format(
            exp_stats["V_loss"][-1] / run_counter,
            exp_stats["V_dispatch_mean"][-1] / run_counter,
            exp_stats["V_topology_mean"][-1] / run_counter,
            exp_stats["V_ineq_mag_mean"][-1] / run_counter,
            exp_stats["V_ineq_mag_max"][-1] / run_counter,
            exp_stats["V_ineq_num_viol_1"][-1] / run_counter,
        )
    )

    return


def parse_data_size(experiment_filepath):
    if len(experiment_filepath) > 1:
        print("Warning: only first experiment file will be plotted")

    exp_stats = {}  # save the stats per run for an experiment
    # read file after line ###
    flag_start = False
    exp_counter = 0
    run_counter = 0
    current_ds = ""
    with open(experiment_filepath[0]) as exp_file:
        while line := exp_file.readline().rstrip():
            if not flag_start:
                if line == "###":
                    flag_start = True
            else:
                exp_counter += 1
                # print(line)
                exp_dir = line[line.find("dir: ") + 13 :]
                exp_filepath = os.path.join(exp_dir, "stats.dict")
                exp_filepath_small = os.path.join(exp_dir, "stats_small.dict")
                if not os.path.exists(exp_filepath) and not os.path.exists(
                    exp_filepath_small
                ):
                    print("Error: experiment line {} not found".format(exp_counter))
                else:
                    exp_filepath_get = (
                        exp_filepath
                        if os.path.exists(exp_filepath)
                        else exp_filepath_small
                    )
                    with open(exp_filepath_get, "rb") as exp_handle:
                        exp_ds = line[line.find("size: ") + 6 : line.find(", run")]
                        exp_run = line[line.find("run: ") + 5 : line.find(", dir")]

                        if not exp_ds == current_ds and not current_ds == "":
                            # plot previous experiment results
                            ## Plotting with new stats (i.e. save results per epoch only)
                            plot_exp_NNsize(exp_stats, run_counter, current_ds)

                            ## print final epoch results:
                            print("\n\n ---------- \nNN size: {}".format(current_ds))
                            print(
                                "\n total loss = {} \n dispatch error = {} \n topology error = {} \n ineq viol"
                                "mean = {} \n ineq viol max = {} \n ineq viol 0.01 = {}".format(
                                    exp_stats["V_loss"][-1] / run_counter,
                                    exp_stats["V_dispatch_mean"][-1] / run_counter,
                                    exp_stats["V_topology_mean"][-1] / run_counter,
                                    exp_stats["V_ineq_mag_mean"][-1] / run_counter,
                                    exp_stats["V_ineq_mag_max"][-1] / run_counter,
                                    exp_stats["V_ineq_num_viol_1"][-1] / run_counter,
                                )
                            )

                            # reset for new experiment
                            current_ds = exp_ds  # update experiment
                            exp_stats = {}  # reset the experiment stats
                            run_counter = 0  # reset run counter

                        if current_ds == "":
                            current_ds = exp_ds

                        # save the stats per run for the experiment
                        stats_dict = pickle.load(exp_handle)  # load the stats
                        # fmt: off
                        dict_agg(exp_stats, 'T_loss', stats_dict["train_loss"])
                        dict_agg(exp_stats, 'V_loss', stats_dict["valid_loss"])
                        dict_agg(exp_stats, 'T_dispatch_mean', stats_dict["train_dispatch_error_mean"])
                        dict_agg(exp_stats, 'V_dispatch_mean', stats_dict["valid_dispatch_error_mean"])
                        dict_agg(exp_stats, 'T_topology_mean', stats_dict["train_topology_error_mean"])
                        dict_agg(exp_stats, 'V_topology_mean', stats_dict["valid_topology_error_mean"])
                        dict_agg(exp_stats, 'V_ineq_num_viol_0', stats_dict["valid_ineq_num_viol_0"])
                        dict_agg(exp_stats, 'V_ineq_num_viol_1', stats_dict["valid_ineq_num_viol_1"])
                        dict_agg(exp_stats, 'V_ineq_mag_max', stats_dict["valid_ineq_max"])
                        dict_agg(exp_stats, 'V_ineq_mag_mean', stats_dict["valid_ineq_mean"])
                        # fmt: on
                        # exp_stats['T_loss'] += stats_dict["train_loss"]
                        # exp_stats['V_loss'] += stats_dict["valid_loss"]
                        # exp_stats['T_dispatch_mean'] += stats_dict["train_dispatch_error_mean"]
                        # exp_stats['V_dispatch_mean'] += stats_dict["valid_dispatch_error_mean"]
                        # exp_stats['T_topology_mean'] += stats_dict["train_topology_error_mean"]
                        # exp_stats['V_topology_mean'] += stats_dict["valid_topology_error_mean"]
                        # exp_stats['V_ineq_num_viol_0'] += stats_dict["valid_ineq_num_viol_0"]
                        # exp_stats['V_ineq_num_viol_1'] += stats_dict["valid_ineq_num_viol_1"]

                        run_counter += 1  # increment run counter
        if flag_start:
            plot_exp_NNsize(exp_stats, run_counter, current_ds)

    if flag_start:
        # add legend and title to the plots
        plt.figure(1)  # train loss
        plt.legend(loc="upper right")
        plt.title("Training loss, mean across samples")
        plt.figure(2)  # valid loss
        plt.legend(loc="upper right")
        plt.title("Validation loss, mean across samples")
        plt.figure(3)  # T & V dispatch error, log scale
        plt.legend(loc="upper right")
        plt.title("Dispatch error, log-y")
        plt.figure(4)  # T & V topology error
        plt.legend(loc="upper right")
        plt.title("Topology error")
        plt.figure(5)  # V ineq error violation
        plt.legend(loc="upper right")
        plt.title("Inequality error violation, 0.001")
        plt.figure(6)  # V ineq error violation
        plt.legend(loc="upper right")
        plt.title("Inequality error violation, 0.01")

        plt.show()  # enter non-interactive mode, and keep plots
    else:
        print("ERROR: File start ### not found")

    return


def parse_pareto_datasize_NN(exp_filepaths):
    # want to plot a pareto front of NN size and number of data points
    datasize_exp_stats = []
    datasize_counter = 0
    for exp_filepath in exp_filepaths:
        NN_exp_stats = []  # save the stats per run for an experiment
        exp_stats = {}
        flag_start = False  # read file after line ###
        run_counter = 0  # to average stats across runs
        current_nn = ""  # to switch to new NN_exp_stats
        if not exp_filepath.find("datasize") == -1:
            current_datasize = int(
                exp_filepath[
                    exp_filepath.find("datasize") + 8 : exp_filepath.find("ver") - 1
                ]
            )
        else:
            current_datasize = 9000  # TODO: fix the default value
        with open(exp_filepath) as exp_file:
            while line := exp_file.readline().rstrip():
                if not flag_start:
                    if line == "###":
                        flag_start = True
                else:
                    exp_dir = line[line.find("dir: ") + 13 :]
                    exp_filepath = os.path.join(exp_dir, "stats.dict")
                    exp_filepath_small = os.path.join(exp_dir, "stats_small.dict")
                    if not os.path.exists(exp_filepath) and not os.path.exists(
                        exp_filepath_small
                    ):
                        print("Error: experiment not found: {}".format(exp_dir))
                    else:
                        exp_filepath_get = (
                            exp_filepath
                            if os.path.exists(exp_filepath)
                            else exp_filepath_small
                        )
                        with open(exp_filepath_get, "rb") as exp_handle:
                            exp_nn = line[line.find("size: ") + 6 : line.find(", run")]
                            exp_run = line[line.find("run: ") + 5 : line.find(", dir")]

                            if not exp_nn == current_nn and not current_nn == "":
                                NN_exp_stats.append(
                                    {
                                        "nn_size": current_nn,
                                        "num_runs": run_counter,
                                        "stats": exp_stats,
                                    }
                                )
                                # reset for new experiment
                                current_nn = exp_nn  # update experiment
                                exp_stats = {}  # reset the experiment stats
                                run_counter = 0  # reset run counter
                            if current_nn == "":
                                current_nn = exp_nn

                            # save the stats per run for the experiment
                            stats_dict = pickle.load(exp_handle)  # load the stats
                            # TODO: get the relevant data
                            # fmt: off
                            dict_agg(exp_stats, 'T_loss', stats_dict["train_loss"])
                            dict_agg(exp_stats, 'V_loss', stats_dict["valid_loss"])
                            dict_agg(exp_stats, 'V_dispatch_mean', stats_dict["valid_dispatch_error_mean"])
                            dict_agg(exp_stats, 'V_topology_mean', stats_dict["valid_topology_error_mean"])
                            dict_agg(exp_stats, 'V_ineq_num_viol_0', stats_dict["valid_ineq_num_viol_0"])
                            dict_agg(exp_stats, 'V_ineq_num_viol_1', stats_dict["valid_ineq_num_viol_1"])
                            dict_agg(exp_stats, 'V_ineq_mag_max', stats_dict["valid_ineq_max"])
                            dict_agg(exp_stats, 'V_ineq_mag_mean', stats_dict["valid_ineq_mean"])
                            # fmt: on
                            run_counter += 1

            if flag_start:
                NN_exp_stats.append(
                    {"nn_size": current_nn, "num_runs": run_counter, "stats": exp_stats}
                )
        datasize_exp_stats.append(
            {"data_size": current_datasize, "all_stats": NN_exp_stats}
        )
        flag_start = False  # reset

    if len(datasize_exp_stats) > 0:
        # plot the pareto front
        plot_pareto_datasize_NN(datasize_exp_stats)

    else:
        print("ERROR: File start ### not found")

    return


def plot_pareto_datasize_NN(exp_stats):
    x_data = [
        str(exp_data["data_size"]) for exp_data in exp_stats
    ]  # number of data points in training
    all_y_data = [
        int(nn_data["nn_size"])
        for exp_data in exp_stats
        for nn_data in exp_data["all_stats"]
    ]
    seen = set()
    repeated_y_data = {x for x in all_y_data if x in seen or (seen.add(x) or False)}
    y_data = sorted(repeated_y_data) * len(x_data)  # number of neurons per hidden layer
    y_data = [str(y) for y in y_data]
    x_data *= len(repeated_y_data)
    num_runs = [
        nn_data["num_runs"]
        for exp_data in exp_stats
        for nn_data in exp_data["all_stats"]
        if int(nn_data["nn_size"]) in repeated_y_data
    ]
    # exp_stats = [{data_size, all_stats=[{nn_size},{nn_size},{nn_size}]}]

    plot_sz = 300
    fig, axs = plt.subplots(1, 5, figsize=(9, 4.5), sharey=True)  # figsize=(9, 3),

    # red = Color("red")
    # colors = list(red.range_to(Color("green"), 10))

    # plot dispatch error
    res = [
        nn_data["stats"]["V_dispatch_mean"]
        for exp_data in exp_stats
        for nn_data in exp_data["all_stats"]
        if int(nn_data["nn_size"]) in repeated_y_data
    ]
    dispatch_error_last = [nn_res[-1] for nn_res in res]
    dispatch = [a / b for a, b in zip(dispatch_error_last, num_runs)]
    dispatch_to_plot = [float(x) * plot_sz / max(dispatch) for x in dispatch]
    # dispatch_colours = 0
    axs[0].scatter(
        x_data, y_data, s=dispatch_to_plot, alpha=0.75
    )  # , c=dispatch_colours
    # axs[0].set_xticks(rotation=45)
    axs[0].set_xticks(
        axs[0].get_xticks(), axs[0].get_xticklabels(), rotation=45, ha="right"
    )
    # find index of maximum
    dispatch_max_ind = dispatch.index(max(dispatch))
    axs[0].annotate(
        str(round(max(dispatch) * 1000) / 1000),
        xy=(x_data[dispatch_max_ind], y_data[dispatch_max_ind]),
        xytext=(-15, 10),
        textcoords="offset points",
    )
    # axs[0].set_title('Dispatch error', wrap=True)
    axs[0].set_title("\n".join(wrap("Dispatch error", 12)))

    # plot topology error
    res = [
        nn_data["stats"]["V_topology_mean"]
        for exp_data in exp_stats
        for nn_data in exp_data["all_stats"]
        if int(nn_data["nn_size"]) in repeated_y_data
    ]
    topology_error_last = [nn_res[-1] for nn_res in res]
    topology = [a / b for a, b in zip(topology_error_last, num_runs)]
    topology_to_plot = [float(x) * plot_sz / max(topology) for x in topology]
    axs[1].scatter(x_data, y_data, s=topology_to_plot, alpha=0.75)  # c=colors,
    axs[1].set_xticks(
        axs[1].get_xticks(), axs[1].get_xticklabels(), rotation=45, ha="right"
    )
    # find index of maximum
    topology_max_ind = topology.index(max(topology))
    axs[1].annotate(
        str(round(max(topology) * 1000) / 10) + "%",
        xy=(x_data[topology_max_ind], y_data[topology_max_ind]),
        xytext=(-15, 10),
        textcoords="offset points",
    )
    # axs[1].set_title('Topology error', wrap=True)
    axs[1].set_title("\n".join(wrap("Topology error", 12)))

    # plot ineq violation mean
    res = [
        nn_data["stats"]["V_ineq_mag_mean"]
        for exp_data in exp_stats
        for nn_data in exp_data["all_stats"]
        if int(nn_data["nn_size"]) in repeated_y_data
    ]
    ineq_mean_error_last = [nn_res[-1] for nn_res in res]
    ineq_mean = [a / b for a, b in zip(ineq_mean_error_last, num_runs)]
    ineq_mean_to_plot = [float(x) * plot_sz / max(ineq_mean) for x in ineq_mean]
    axs[2].scatter(x_data, y_data, s=ineq_mean_to_plot, alpha=0.75)  # c=colors,
    axs[2].set_xticks(
        axs[2].get_xticks(), axs[2].get_xticklabels(), rotation=45, ha="right"
    )
    # find index of maximum
    ineq_mean_max_ind = ineq_mean.index(max(ineq_mean))
    axs[2].annotate(
        str(round(max(ineq_mean) * 5680)) + " kVA",
        xy=(x_data[ineq_mean_max_ind], y_data[ineq_mean_max_ind]),
        xytext=(-15, 10),
        textcoords="offset points",
    )
    # axs[2].set_title('Ineq violation mean', wrap=True)
    axs[2].set_title("\n".join(wrap("Ineq viol mean", 12)))

    # plot ineq violation max
    res = [
        nn_data["stats"]["V_ineq_mag_max"]
        for exp_data in exp_stats
        for nn_data in exp_data["all_stats"]
        if int(nn_data["nn_size"]) in repeated_y_data
    ]
    ineq_max_error_last = [nn_res[-1] for nn_res in res]
    ineq_max = [a / b for a, b in zip(ineq_max_error_last, num_runs)]
    ineq_max_to_plot = [float(x) * plot_sz / max(ineq_max) for x in ineq_max]
    axs[3].scatter(x_data, y_data, s=ineq_max_to_plot, alpha=0.75)  # c=colors,
    axs[3].set_xticks(
        axs[3].get_xticks(), axs[3].get_xticklabels(), rotation=45, ha="right"
    )
    # find index of maximum
    ineq_max_max_ind = ineq_max.index(max(ineq_max))
    axs[3].annotate(
        str(round(max(ineq_max) * 5680)) + " kVA",
        xy=(x_data[ineq_max_max_ind], y_data[ineq_max_max_ind]),
        xytext=(-12, 10),
        textcoords="offset points",
    )
    # axs[3].set_title('Ineq violation max', wrap=True)
    axs[3].set_title("\n".join(wrap("Ineq viol max", 12)))

    # plot ineq violation at level 1
    res = [
        nn_data["stats"]["V_ineq_num_viol_1"]
        for exp_data in exp_stats
        for nn_data in exp_data["all_stats"]
        if int(nn_data["nn_size"]) in repeated_y_data
    ]
    ineq_v1_error_last = [nn_res[-1] for nn_res in res]
    ineq_v1 = [a / b for a, b in zip(ineq_v1_error_last, num_runs)]
    ineq_v1_to_plot = [float(x) * plot_sz / max(ineq_v1) for x in ineq_v1]
    axs[4].scatter(x_data, y_data, s=ineq_v1_to_plot, alpha=0.75)  # c=colors,
    axs[4].set_xticks(
        axs[4].get_xticks(), axs[4].get_xticklabels(), rotation=45, ha="right"
    )
    # find index of maximum
    ineq_v1_max_ind = ineq_v1.index(max(ineq_v1))
    axs[4].annotate(
        str(round(max(ineq_v1) / 1465 * 100)) + "%",
        xy=(x_data[ineq_v1_max_ind], y_data[ineq_v1_max_ind]),
        xytext=(-10, 10),
        textcoords="offset points",
    )
    # axs[4].set_title('Num ineq violations > 0.01', wrap=True)
    axs[4].set_title("\n".join(wrap("Num ineq viol > 0.01", 12)))

    fig.savefig("pareto_experiment_newlabels.eps", format="eps")
    return


if __name__ == "__main__":
    print("Plotting ------")
    main()
