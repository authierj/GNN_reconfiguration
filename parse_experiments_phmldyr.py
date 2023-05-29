# file to parse results from experiments and produce plots
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils_JA import dict_agg

def main():
    # exp_names = ["GatedSwitchGNN_globalMLP_lr_test","GatedSwitchGNN_globalMLP_numLayers_test", "GatedSwitchGNN_globalMLP_hiddenFeatures_test"]
    # exp_names = ["GatedSwitchGNN_lr_test","GatedSwitchGNN_numLayers_test", "GatedSwitchGNN_hiddenFeatures_test"]
    # exp_names = ["GCN_Global_MLP_reduced_model_numLayers_test", "GCN_Global_MLP_reduced_model_hiddenFeatures_test"]
    exp_names = ["testing_test"]
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
    # best_topo = np.zeros(500)
    # worst_topo = np.zeros(500)
    current_nn = ""
    n = 0
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
                        exp_nn = line[
                            line.find("testing_test") + 15 : line.find("/v") - 1
                        ]
                        exp_nn = "GatedGNN" + exp_nn
                        # exp_nn = line[line.find("lr: ") : line.find(", run")]
                        # exp_nn = line[line.find("dir: ") + 13 : -3]
                        exp_run = line[line.find("run: ") + 5 : line.find(", dir")]

                        if exp_nn != current_nn and current_nn != "":
                            n += 1
                            # plot previous experiment results
                            ## Plotting with new stats (i.e. save results per epoch only)
                            """
                            # fmt: off
                            exp_stats["T_topology_best"] = best_topo 
                            exp_stats["T_topology_worst"] = worst_topo
                            exp_stats["T_topology_best"] = best_topo 
                            exp_stats["T_topology_worst"] = worst_topo
                            exp_stats["T_loss_var"] = np.var(exp_stats["T_loss_var"], axis=0)
                            exp_stats["V_loss_var"] = np.var(exp_stats["V_loss_var"], axis=0)
                            exp_stats["T_dispatch_mean_var"] = np.var(exp_stats["T_dispatch_mean_var"], axis=0)
                            exp_stats["V_dispatch_mean_var"] = np.var(exp_stats["V_dispatch_mean_var"], axis=0)
                            exp_stats["T_topology_mean_var"] = np.var(exp_stats["T_topology_mean_var"], axis=0)
                            exp_stats["V_topology_mean_var"] = np.var(exp_stats["V_topology_mean_var"], axis=0)
                            # fmt: on
                            # plt.figure(n*10)
                            # p_switches = exp_stats["V_pswitch"]
                            # for i in range(p_switches.shape[1]):
                            #     plt.plot(p_switches[:1000,i], label=f"switch{i}", color=f"C{i}")
                            #     plt.plot(p_switches[1000:,i], "--", label=f"switch{i}", color=f"C{i}")

                            # plt.title(f'{n}')
                            """
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
                        dict_agg(exp_stats, 'TE_loss', stats_dict["test_loss"], op="sum")

                        dict_agg(exp_stats, 'T_dispatch_mean', stats_dict["train_dispatch_error_mean"], op="sum")
                        dict_agg(exp_stats, 'V_dispatch_mean', stats_dict["valid_dispatch_error_mean"], op="sum")
                        dict_agg(exp_stats, 'TE_dispatch_mean', stats_dict["test_dispatch_error_mean"], op="sum")
                        
                        dict_agg(exp_stats, 'T_voltage_mean', stats_dict["train_voltage_error_mean"], op="sum")
                        dict_agg(exp_stats, 'V_voltage_mean', stats_dict["valid_voltage_error_mean"], op="sum")
                        dict_agg(exp_stats, 'TE_voltage_mean', stats_dict["test_voltage_error_mean"], op="sum")

                        dict_agg(exp_stats, 'T_topology_mean', stats_dict["train_topology_error_mean"], op="sum")
                        dict_agg(exp_stats, 'V_topology_mean', stats_dict["valid_topology_error_mean"], op="sum")
                        dict_agg(exp_stats, 'TE_topology_mean', stats_dict["test_topology_error_mean"], op="sum")
                        
                        dict_agg(exp_stats, 'V_ineq_num_viol_0', stats_dict["valid_ineq_num_viol_0"], op="sum")
                        dict_agg(exp_stats, 'V_ineq_num_viol_1', stats_dict["valid_ineq_num_viol_1"], op="sum")
                        dict_agg(exp_stats, 'V_ineq_mag_max', stats_dict["valid_ineq_max"], op="sum")
                        dict_agg(exp_stats, 'V_ineq_mag_mean', stats_dict["valid_ineq_mean"], op="sum")
                        dict_agg(exp_stats, 'TE_ineq_num_viol_0', stats_dict["test_ineq_num_viol_0"], op="sum")
                        dict_agg(exp_stats, 'TE_ineq_num_viol_1', stats_dict["test_ineq_num_viol_1"], op="sum")
                        dict_agg(exp_stats, 'TE_ineq_mag_max', stats_dict["test_ineq_max"], op="sum")
                        dict_agg(exp_stats, 'TE_ineq_mag_mean', stats_dict["test_ineq_mean"], op="sum")
                        
                        dict_agg(exp_stats, 'T_opt_gap', stats_dict["valid_opt_gap"], op="vstack")
                        dict_agg(exp_stats, 'V_opt_gap', stats_dict["valid_opt_gap"], op="vstack")
                        dict_agg(exp_stats, 'TE_opt_gap', stats_dict["test_opt_gap"], op="vstack")
                        
                        """
                        dict_agg(exp_stats, 'T_topology_max', stats_dict["train_topology_error_max"], op="sum")
                        dict_agg(exp_stats, 'T_topology_min', stats_dict["train_topology_error_min"], op="sum")
                        dict_agg(exp_stats, 'T_loss_var', stats_dict["train_loss"].copy(), op="vstack")
                        dict_agg(exp_stats, 'V_loss_var', stats_dict["valid_loss"].copy(), op="vstack")
                        dict_agg(exp_stats, 'T_dispatch_mean_var', stats_dict["train_dispatch_error_mean"].copy(), op="vstack")
                        dict_agg(exp_stats, 'V_dispatch_mean_var', stats_dict["valid_dispatch_error_mean"].copy(), op="vstack")
                        dict_agg(exp_stats, 'T_topology_mean_var', stats_dict["train_topology_error_mean"].copy(), op="vstack")
                        dict_agg(exp_stats, 'V_topology_mean_var', stats_dict["valid_topology_error_mean"].copy(), op="vstack")
                        dict_agg(exp_stats, 'T_loss_var', stats_dict["train_loss"].copy(), op="vstack")
                        dict_agg(exp_stats, 'V_loss_var', stats_dict["valid_loss"].copy(), op="vstack")
                        dict_agg(exp_stats, 'T_dispatch_mean_var', stats_dict["train_dispatch_error_mean"].copy(), op="vstack")
                        dict_agg(exp_stats, 'V_dispatch_mean_var', stats_dict["valid_dispatch_error_mean"].copy(), op="vstack")
                        dict_agg(exp_stats, 'T_topology_mean_var', stats_dict["train_topology_error_mean"].copy(), op="vstack")
                        dict_agg(exp_stats, 'V_topology_mean_var', stats_dict["valid_topology_error_mean"].copy(), op="vstack")
                        dict_agg(exp_stats, 'V_pswitch', stats_dict["valid_pswitch"], op="vstack")
                        """
                        # fmt: on
                        
                        if run_counter == 0:
                            best_topo = stats_dict["train_topology_error_mean"].copy()
                            worst_topo = stats_dict["train_topology_error_mean"].copy()
                        elif (
                            stats_dict["train_topology_error_mean"][-1] < best_topo[-1]
                        ):
                            best_topo = stats_dict["train_topology_error_mean"].copy()
                        elif (
                            stats_dict["train_topology_error_mean"][-1] > worst_topo[-1]
                        ):
                            worst_topo = stats_dict["train_topology_error_mean"].copy()
                        if run_counter == 0:
                            best_topo = stats_dict["train_topology_error_mean"].copy()
                            worst_topo = stats_dict["train_topology_error_mean"].copy()
                        elif (
                            stats_dict["train_topology_error_mean"][-1] < best_topo[-1]
                        ):
                            best_topo = stats_dict["train_topology_error_mean"].copy()
                        elif (
                            stats_dict["train_topology_error_mean"][-1] > worst_topo[-1]
                        ):
                            worst_topo = stats_dict["train_topology_error_mean"].copy()
                        run_counter += 1  # increment run counter

        if flag_start:
            """
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
            plt.figure(80)
            p_switches = exp_stats["V_pswitch"]
            for i in range(p_switches.shape[1]):
                plt.plot(p_switches[:1000,i], label=f"switch{i}", color=f"C{i}")
                plt.plot(p_switches[1000:,i], "--", label=f"switch{i}", color=f"C{i}")

            plt.title('8')
            """
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
        plt.figure(4)  # T & V dispatch error, log scale
        plt.legend(loc="upper right")
        plt.title(gnn + ", Voltage error, log-y")
        plt.figure(5)  # T & V topology error
        plt.legend(loc="upper right")
        plt.title(gnn + ", Topology error")
        plt.figure(6)  # V ineq error violation
        plt.legend(loc="upper right")
        plt.title(gnn + ", Inequality error violation, 0.001")
        plt.figure(7)  # V ineq error violation
        plt.legend(loc="upper right")
        plt.title(gnn + ", Inequality error violation, 0.01")
        # plt.figure(7)  # V ineq error violation
        # plt.legend(loc="upper right")
        # plt.title(gnn + ", Switching probability")
        # plt.show()  # enter non-interactive mode, and keep plots
    else:
        print("ERROR: File start ### not found")
    return


def plot_exp_NNsize(exp_stats, run_counter, current_nn, exp_counter):
    x = np.arange(exp_stats["T_loss"].shape[0])
    plt.figure(1)  # train loss
    plt.yscale("log")
    plt.plot(
        exp_stats["T_loss"] / run_counter,
        label="current_nn",
        color=f"C{exp_counter}",
    )
    # plt.fill_between(
    #     x=x,
    #     y1=-np.sqrt(exp_stats["T_loss_var"]) + exp_stats["T_loss"] / run_counter,
    #     y2=np.sqrt(exp_stats["T_loss_var"]) + exp_stats["T_loss"] / run_counter,
    #     color=f"C{exp_counter}",
    #     alpha=0.2,
    # )
    plt.figure(2)  # valid loss
    plt.yscale("log")
    plt.plot(
        exp_stats["V_loss"] / run_counter,
        color=f"C{exp_counter}",
        label="validation",
    )
    plt.plot(
        exp_stats["TE_loss"] / run_counter,
        color=f"C{exp_counter+1}",
        label="test",
    )
    plt.ylim([5 * 1e0, 1e4])
    # plt.fill_between(
    #     x=x,
    #     y1=-np.sqrt(exp_stats["V_loss_var"]) + exp_stats["V_loss"] / run_counter,
    #     y2=np.sqrt(exp_stats["V_loss_var"]) + exp_stats["V_loss"] / run_counter,
    #     color=f"C{exp_counter}",
    #     alpha=0.2,
    # )
    plt.figure(3)  # T & V dispatch error, log scale
    plt.yscale("log")
    plt.plot(
        exp_stats["T_dispatch_mean"] / run_counter,
        color=f"C{exp_counter}",
    )
    plt.plot(
        exp_stats["V_dispatch_mean"] / run_counter, "--", color=f"C{exp_counter}"
    )  #  + '-V'
    plt.plot(
        exp_stats["TE_dispatch_mean"] / run_counter,
        color=f"C{exp_counter+1}",
        label="test",
    )  #  + '-V'
    
    plt.figure(4)  # T & V dispatch error, log scale
    plt.yscale("log")
    plt.plot(
        exp_stats["T_voltage_mean"] / run_counter,
        color=f"C{exp_counter}",
    )
    plt.plot(
        exp_stats["V_voltage_mean"] / run_counter, "--", color=f"C{exp_counter}"
    )  #  + '-V'
    plt.plot(
        exp_stats["TE_voltage_mean"] / run_counter,
        color=f"C{exp_counter+1}",
        label="test",
    )  #  + '-V'

    
    plt.figure(5)  # T & V topology error
    plt.plot(
        exp_stats["T_topology_mean"] / run_counter,
        color=f"C{exp_counter}"
    )
    plt.plot(
        exp_stats["V_topology_mean"] / run_counter, linestyle='dotted',
        color=f"C{exp_counter}",
    )
    plt.plot(
        exp_stats["TE_topology_mean"] / run_counter,
        color=f"C{exp_counter+1}", label="test",
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
    # plt.plot(exp_stats["T_topology_best"],'-.', color=f"C{exp_counter}")
    # plt.plot(exp_stats["T_topology_worst"],'--', color=f"C{exp_counter}")

    # plt.ylim([0,4/7+0.01])

    plt.figure(6)  # V ineq error violation
    plt.plot(exp_stats["V_ineq_num_viol_0"] / run_counter)
    plt.figure(7)  # V ineq error violation
    plt.plot(exp_stats["V_ineq_num_viol_1"] / run_counter)

    # print final epoch results:
    print("\n\n ---------- \nNN size: {} \n".format(current_nn))
    print("validation numbers")
    print(
        "\n total loss = {} \n dispatch error = {} \n voltage error = {} \n topology error = {} \n ineq viol"
        "mean = {} \n ineq viol max = {} \n ineq viol 0.01 = {} \n opt. gap = {}\n".format(
            exp_stats["V_loss"][-1] / run_counter,
            exp_stats["V_dispatch_mean"][-1] / run_counter,
            exp_stats["V_voltage_mean"][-1] / run_counter,
            exp_stats["V_topology_mean"][-1] / run_counter,
            exp_stats["V_ineq_mag_mean"][-1] / run_counter,
            exp_stats["V_ineq_mag_max"][-1] / run_counter,
            exp_stats["V_ineq_num_viol_1"][-1] / run_counter,
            exp_stats["V_opt_gap"][-1] / run_counter,
        )
    )
    print("\ntest numbers")
    print(
        "\n total loss = {} \n dispatch error = {} \n voltage error = {} \n topology error = {} \n ineq viol"
        "mean = {} \n ineq viol max = {} \n ineq viol 0.01 = {} \n opt. gap = {}\n".format(
            exp_stats["TE_loss"][-1] / run_counter,
            exp_stats["TE_dispatch_mean"][-1] / run_counter,
            exp_stats["TE_voltage_mean"][-1] / run_counter,
            exp_stats["TE_topology_mean"][-1] / run_counter,
            exp_stats["TE_ineq_mag_mean"][-1] / run_counter,
            exp_stats["TE_ineq_mag_max"][-1] / run_counter,
            exp_stats["TE_ineq_num_viol_1"][-1] / run_counter,
            exp_stats["TE_opt_gap"][-1] / run_counter,
        )
    )

if __name__ == "__main__":
    print("Plotting ------")
    main()
