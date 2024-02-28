import numpy as np
from matplotlib import pyplot as plt, colors, axes
from matplotlib.cm import ScalarMappable


def plot_single(data: dict, errors: dict, data_error: dict, score_2D_dct: dict, validity_domain_dct: dict,
                novelty_2D_dct: dict, threshold: float, save_name: str, save_plot: bool):
    """ Plots 2D UseCases

    Parameters
    ----------
    data: dict
        from 1_config_ann_train.py
    errors: dict
        from 1_config_ann_train.py
    data_error: dict
        from 2_config_score_ann.py
    score_2D_dct: dict
        from 2_config_score_ann.py
    validity_domain_dct: dict
        from 3_config_validity_domain.py
    novelty_2D_dct: dict
        from 3_config_validity_domain.py
    threshold: float
        from 5_train_detector.py
    save_name: str
        Name of the plot
    save_plot: str
        If True, plot is saved to disc. If False, plot is shown
    """
    # Define colors
    green = (87 / 255, 171 / 255, 39 / 255)
    darkgrey = (78 / 255, 79 / 255, 80 / 255)
    red = (221 / 255, 64 / 255, 45 / 255)
    black = (0, 0, 0)
    blue = (0 / 255, 84 / 255, 159 / 255)
    petrol = (0 / 255, 97 / 255, 101 / 255)
    violett = (97 / 255, 33 / 255, 88 / 255)

    # Subplots
    fig, axs = plt.subplots(1, 1, figsize=(5.4, 4.1))
    plt.subplots_adjust(left=0.11, right=0.89, bottom=0.12, top=0.85, wspace=0.05, hspace=0.2)

    # Color Norm
    divnorm = colors.TwoSlopeNorm(
        vmin=min(np.amin(errors['train_error']), np.amin(data_error['errors']), np.amin(errors['val_error']),
                 np.amin(errors['test_error'])), vcenter=validity_domain_dct['error_threshold'],
        vmax=4.01,
        # vmax=max(np.amax(errors['train_error']), np.amax(data_error['errors']), np.amax(errors['val_error']),
        #          np.amax(errors['test_error']))
    )

    # Plot data
    axs.scatter(data['non_available_data'].x_remaining[:, 0], data['non_available_data'].x_remaining[:, 1], c=data_error['errors'], norm=divnorm,
                cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red], N=256), marker='.',
                label='Evaluation Data')
    # Plot training
    axs.scatter(data['available_data'].xTrain[:, 0], data['available_data'].xTrain[:, 1], c=errors['train_error'],
                norm=divnorm,
                cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red], N=256), edgecolors=black,
                linewidths=1.5, label='Training Data')
    # Plot validation and testing
    axs.scatter(data['available_data'].xValid[:, 0], data['available_data'].xValid[:, 1], c=errors['val_error'],
                norm=divnorm,
                cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red], N=256), edgecolors=blue,
                linewidths=1.5, label='Validation Data')
    axs.scatter(data['available_data'].xTest[:, 0], data['available_data'].xTest[:, 1], c=errors['test_error'],
                norm=divnorm,
                cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red], N=256), edgecolors=blue,
                linewidths=1.5)

    # Colorbar
    ticks = [0.01, 0.05, 0.1, 2, 4]
    cb = plt.colorbar(ScalarMappable(norm=divnorm,
                                     cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red],
                                                                                   N=256)),
                      ax=axs,
                      fraction=0.05,
                      # orientation='horizontal',
                      ticks=[round(i, 2) for i in ticks],
                      label='Absolute Prediction Error of ANN in kW'
                      )
    # cb.ax.set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Validity domain
    c1 = axs.contour(score_2D_dct['var1_meshgrid'], score_2D_dct['var2_meshgrid'], score_2D_dct['error_on_mesh'],
                     levels=[validity_domain_dct['error_threshold']],
                     linewidths=2.5, colors=[petrol])
    h1, _ = c1.legend_elements()

    # Classifier decision boundary
    c2 = axs.contour(novelty_2D_dct['var1_meshgrid'], novelty_2D_dct['var2_meshgrid'], novelty_2D_dct['contour_ns_scores'],
                     levels=[threshold], linewidths=2.5, colors=[violett], linestyles='solid')
    h2, _ = c2.legend_elements()

    # Legend
    handles, labels = axs.get_legend_handles_labels()
    handles.append(h1[0])
    labels.append('Validity Domain')
    handles.append(h2[0])
    labels.append('Extrapolation Boundary')
    handles[0], handles[1], handles[2], handles[3],  handles[4] = handles[1], handles[3], handles[2], handles[4], handles[0]
    labels[0], labels[1], labels[2], labels[3], labels[4] = labels[1], labels[3], labels[2],labels[4], labels[0]
    axs.legend(handles, labels, bbox_to_anchor=(0.55, 1.02), ncol=3, loc='lower center', columnspacing=0.3)

    # Axs limits
    axs.set_xlim(-7.5, 20)
    axs.set_ylim(0, 4.5)

    # Axs labels
    axs.set_xlabel(r'$\mathrm{T}_\mathrm{amb}$ in °C')
    axs.set_ylabel(r'$\mathrm{P}_\mathrm{el}$ in kW')

    # Plotting
    # plt.tight_layout()
    if not save_plot:
        plt.show()
    else:
        plt.savefig('plots\\' + save_name, bbox_inches='tight')


def _plot_subplot(data: dict, errors: dict, data_error: dict, score_2D_dct: dict, validity_domain_dct: dict,
                  novelty_2D_dct: dict, threshold: float, axs: axes.Axes, title: str):
    """ Plots 2D UseCases

    Parameters
    ----------
    data: dict
        from 1_config_ann_train.py
    errors: dict
        from 1_config_ann_train.py
    data_error: dict
        from 2_config_score_ann.py
    score_2D_dct: dict
        from 2_config_score_ann.py
    validity_domain_dct: dict
        from 3_config_validity_domain.py
    novelty_2D_dct: dict
        from 3_config_validity_domain.py
    threshold: float
        from 5_train_detector.py
    """
    # Define colors
    green = (87 / 255, 171 / 255, 39 / 255)
    darkgrey = (78 / 255, 79 / 255, 80 / 255)
    red = (221 / 255, 64 / 255, 45 / 255)
    black = (0, 0, 0)
    blue = (0 / 255, 84 / 255, 159 / 255)
    petrol = (0 / 255, 97 / 255, 101 / 255)
    violett = (97 / 255, 33 / 255, 88 / 255)

    # Color Norm
    divnorm = colors.TwoSlopeNorm(
        vmin=min(np.amin(errors['train_error']), np.amin(data_error['errors']), np.amin(errors['val_error']),
                 np.amin(errors['test_error'])), vcenter=validity_domain_dct['error_threshold'],
        # vmax=max(np.amax(errors['train_error']), np.amax(data_error['errors']), np.amax(errors['val_error']),
        #          np.amax(errors['test_error']))
        vmax=8.01
    )

    markersize = 20

    # Plot data
    axs.scatter(data['non_available_data'].x_remaining[:, 0], data['non_available_data'].x_remaining[:, 1], c=data_error['errors'], norm=divnorm,
                cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red], N=256), marker='.',
                label='Evaluation Data', alpha=1)
    # Plot training
    axs.scatter(data['available_data'].xTrain[:, 0], data['available_data'].xTrain[:, 1], c=errors['train_error'],
                norm=divnorm,
                cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red], N=256), edgecolors=black,
                linewidths=1.5, label='Training Data', s=markersize)
    # Plot validation and testing
    axs.scatter(data['available_data'].xValid[:, 0], data['available_data'].xValid[:, 1], c=errors['val_error'],
                norm=divnorm,
                cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red], N=256), edgecolors=blue,
                linewidths=1.5, label='Validation Data', s=markersize)
    axs.scatter(data['available_data'].xTest[:, 0], data['available_data'].xTest[:, 1], c=errors['test_error'],
                norm=divnorm,
                cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red], N=256), edgecolors=blue,
                linewidths=1.5, s=markersize)

    # Colorbar
    # cb = plt.colorbar(ScalarMappable(norm=divnorm,
    #                                  cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red],
    #                                                                                N=256)),
    #                   ax=axs,
    #                   fraction=0.05,
    #                   )
    # cb.ax.set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Validity domain
    c1 = axs.contour(score_2D_dct['var1_meshgrid'], score_2D_dct['var2_meshgrid'], score_2D_dct['error_on_mesh'],
                     levels=[validity_domain_dct['error_threshold']],
                     linewidths=2.5, colors=[petrol])
    h1, _ = c1.legend_elements()

    # Classifier decision boundary
    c2 = axs.contour(novelty_2D_dct['var1_meshgrid'], novelty_2D_dct['var2_meshgrid'], novelty_2D_dct['contour_ns_scores'],
                     levels=[threshold], linewidths=2.5, colors=[violett], linestyles='solid')
    h2, _ = c2.legend_elements()

    # Legend
    handles, labels = axs.get_legend_handles_labels()
    handles.append(h1[0])
    labels.append('Validity Domain')
    handles.append(h2[0])
    labels.append('Extrapolation Boundary')

    # Axs limits
    axs.set_xlim(-7.5, 20)
    axs.set_ylim(0, 4.5)

    axs.set_title(title)
    print(title)
    print(threshold)

    # Axs labels
    # axs.set_xlabel(r'$\mathrm{T}_\mathrm{amb}$ in °C')
    # axs.set_ylabel(r'$\mathrm{P}_\mathrm{el}$ in kW')

    return labels, handles, divnorm


def plot_multiple(data: list, errors: list, data_error: list, score_2D_dct: list, validity_domain_dct: list,
                  novelty_2D_dct: list, threshold: list, save_name: str, save_plot: bool, title: list):
    n = len(data)
    assert n == 5

    # Define colors
    green = (87 / 255, 171 / 255, 39 / 255)
    darkgrey = (78 / 255, 79 / 255, 80 / 255)
    red = (221 / 255, 64 / 255, 45 / 255)

    fig, axs = plt.subplots(5, 2, figsize=(5.4, 7.2), height_ratios=[1, 0, 1, 0, 1])
    plt.subplots_adjust(left=0.075, right=0.98, bottom=0.065, top=0.965, wspace=0.05, hspace=0.2)

    handles = None
    labels = None
    divnorm = None
    for row in range(0, 5):
        for column in range(0, 2):
            if row == 1 or row == 3:
                axs[row, column].remove()
            else:
                if row == 0:
                    indx = column
                elif row == 2:
                    indx = column + 2
                elif row == 4:
                    indx = column + 4
                if indx < 5:
                    labels, handles, divnorm = _plot_subplot(data[indx], errors[indx], data_error[indx],
                                                             score_2D_dct[indx],
                                                             validity_domain_dct[indx], novelty_2D_dct[indx],
                                                             threshold[indx],
                                                             axs[row, column], title=title[indx])
                    if indx != 3 and indx != 4:  # and indx != 2:
                        axs[row, column].set_xticklabels([])
                    else:
                        axs[row, column].set_xlabel(r'$\mathrm{T}_\mathrm{amb}$ in °C', labelpad=0)

                    if indx != 0 and indx != 2 and indx != 4:
                        axs[row, column].set_yticklabels([])
                    else:
                        axs[row, column].set_ylabel(r'$\mathrm{P}_\mathrm{el}$ in kW', labelpad=0)
                else:
                    axs[row, column].axis('off')

    handles[0], handles[1], handles[2] = handles[1], handles[2], handles[0]
    labels[0], labels[1], labels[2] = labels[1], labels[2], labels[0]

    axs[4, 1].legend(handles, labels)
    #ticks = np.linspace(divnorm.vmin + 0.01, divnorm.vcenter, 3).tolist() + np.linspace(divnorm.vcenter, divnorm.vmax - 0.01, 3).tolist()
    ticks = [0.01, 0.04, 0.08, 4, 8]
    cb = plt.colorbar(ScalarMappable(norm=divnorm,
                                     cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red],
                                                                                   N=256)),
                      ax=axs[4, 1],
                      fraction=0.05,
                      orientation='horizontal',
                      ticks=[round(i, 2) for i in ticks],
                      label='Absolute Prediction Error \n of ANN in kW'
                      )
    cb.ax.xaxis.set_label_position('top')

    # plt.tight_layout()
    if not save_plot:
        plt.show()
    else:
        plt.savefig('plots\\' + save_name)


def plot_multiple_3(data: list, errors: list, data_error: list, score_2D_dct: list, validity_domain_dct: list,
                  novelty_2D_dct: list, threshold: list, save_name: str, save_plot: bool, title: list):
    n = len(data)
    assert n == 3

    # Define colors
    green = (87 / 255, 171 / 255, 39 / 255)
    darkgrey = (78 / 255, 79 / 255, 80 / 255)
    red = (221 / 255, 64 / 255, 45 / 255)

    fig, axs = plt.subplots(3, 2, figsize=(5.4, 4.8), height_ratios=[1, 0, 1])
    plt.subplots_adjust(left=0.075, right=0.98, bottom=0.09, top=0.95, wspace=0.05, hspace=0.2)

    handles = None
    labels = None
    divnorm = None
    for row in range(0, 3):
        for column in range(0, 2):
            if row == 1 or row == 3:
                axs[row, column].remove()
            else:
                if row == 0:
                    indx = column
                elif row == 2:
                    indx = column + 2
                # elif row == 4:
                #     indx = column + 4
                if indx < 3:
                    labels, handles, divnorm = _plot_subplot(data[indx], errors[indx], data_error[indx],
                                                             score_2D_dct[indx],
                                                             validity_domain_dct[indx], novelty_2D_dct[indx],
                                                             threshold[indx],
                                                             axs[row, column], title=title[indx])
                    if indx != 1 and indx != 2:  # and indx != 2:
                        axs[row, column].set_xticklabels([])
                    else:
                        axs[row, column].set_xlabel(r'$\mathrm{T}_\mathrm{amb}$ in °C', labelpad=0)

                    if indx != 0 and indx != 2 and indx != 4:
                        axs[row, column].set_yticklabels([])
                    else:
                        axs[row, column].set_ylabel(r'$\mathrm{P}_\mathrm{el}$ in kW', labelpad=0)
                else:
                    axs[row, column].axis('off')

    handles[0], handles[1], handles[2] = handles[1], handles[2], handles[0]
    labels[0], labels[1], labels[2] = labels[1], labels[2], labels[0]

    axs[2, 1].legend(handles, labels)
    #ticks = np.linspace(divnorm.vmin + 0.01, divnorm.vcenter, 3).tolist() + np.linspace(divnorm.vcenter, divnorm.vmax - 0.01, 3).tolist()
    ticks = [0.01, 0.05, 0.1, 4, 8]
    cb = plt.colorbar(ScalarMappable(norm=divnorm,
                                     cmap=colors.LinearSegmentedColormap.from_list('gwr', [green, darkgrey, red],
                                                                                   N=256)),
                      ax=axs[2, 1],
                      fraction=0.05,
                      orientation='horizontal',
                      ticks=[round(i, 2) for i in ticks],
                      label='Absolute Prediction Error \n of ANN in kW'
                      )
    cb.ax.xaxis.set_label_position('top')

    # plt.tight_layout()
    if not save_plot:
        plt.show()
    else:
        plt.savefig('plots\\' + save_name)

def plot_bar_plot(bars, clf_labels, bars_ideal=None, labels=None, ylabel=None):

    blue = (0 / 255, 84 / 255, 159 / 255)
    green = (87 / 255, 171 / 255, 39 / 255)
    violett = (97 / 255, 33 / 255, 88 / 255)
    black = (0, 0, 0)
    magenta = (227 / 255, 0 / 255, 102 / 255)
    orange = (246 / 255, 168 / 255, 0 / 255)
    darkred = (114 / 255, 29 / 255, 19 / 255)
    color_list = [blue, black, green, darkred, orange, magenta, violett]

    n_uc = bars.shape[1]
    n_clfs = bars.shape[0]
    bar_width = 0.8 / n_clfs
    r1 = np.arange(n_uc)

    fig = plt.figure(figsize=(5.4, 4))
    plt.ylim(0, 1)

    for i in range(n_clfs):
        r = [x + bar_width * i for x in r1]
        bar = bars[i, :]
        if bars_ideal is not None:
            bar_ideal = bars_ideal[i, :]
            yerr = np.zeros((1,n_uc))
            yerr[0,:] = bar_ideal - bar
            # for j in range(len(bar)):
            #     plt.hlines(bar[j] + yerr[0][j], r[j] - bar_width/2, r[j] + bar_width/2, color=color_list[i])
            #     plt.vlines(r[j] - bar_width/2 * 0.93, bar[j], bar[j] + yerr[0][j], color=color_list[i])
            #     plt.vlines(r[j] + bar_width / 2 * 0.93, bar[j], bar[j] + yerr[0][j], color=color_list[i])
            plt.bar(r, bar_ideal,
                    # bottom=bar,
                    hatch='//', width=bar_width * 0.97, color='none', edgecolor=color_list[i])

        plt.bar(r, bar, width=bar_width * 0.97, color=color_list[i], edgecolor=color_list[i], label=clf_labels[i])

    if labels is None:
        plt.xticks([x + bar_width * (n_clfs-1) / 2 for x in r1])
    else:
        plt.xticks([x + bar_width * (n_clfs-1) / 2 for x in r1], labels)

    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('plots\\barplot.pdf', bbox_inches='tight')
    plt.show()

