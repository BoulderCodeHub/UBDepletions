import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm, colors
import seaborn as sns
from scipy.stats import gaussian_kde
import os

def label_historical_regressions(ax_to_label, r2_vals, color_list, label_list):
  ax_to_label.set_ylabel('Unregulated Powell Inflow, 1963-2020 (obs, maf/yr)')
  ax_to_label.set_yticks([2500000, 5000000, 7500000, 10000000, 12500000, 15000000, 17500000, 20000000, 22500000])
  ax_to_label.set_yticklabels([2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5])
  ax_to_label.set_xlabel('Lees Ferry Natural Flow, 1963-2020 (maf/yr)')
  ax_to_label.set_xticks([5000000, 7500000, 10000000, 12500000, 15000000, 17500000, 20000000, 22500000, 25000000])
  ax_to_label.set_xticklabels([5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25])
  x_vals = [10000000, 20000000]
  y_vals = [18000000, 5000000]
  for txt_idx in range(0, len(r2_vals)):
    ax_to_label.text(x_vals[txt_idx], y_vals[txt_idx], 'R*2: ' + str(int(r2_vals[txt_idx] * 1000) / 1000.0), fontsize = 14, horizontalalignment='center', verticalalignment='center')
  ax_to_label.set_xlim([5000000, 25000000])
  ax_to_label.set_ylim([2500000, 22500000])
  custom_lines = []
  label_lines = []
  for color_use, label_use in zip(color_list, label_list):
    custom_lines.append(Line2D([0], [0], color = color_use,  lw = 2.5))
    label_lines.append(label_use)
  ax_to_label.legend(custom_lines, label_lines, loc = 'upper left', prop={'size': 12})
  return ax_to_label

def label_error_plots(ax_label, cbar_label):
  ax_label.set_ylabel('Annual UB Demands (maf)')
  ax_label.set_yticks([4000000, 5000000, 6000000])
  ax_label.set_yticklabels([4, 5, 6])  
  ax_label.set_xlabel('Lees Ferry Natural Flow')
  ax_label.set_xticks([5000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000, 40000000])
  ax_label.set_xticklabels([5, 10, 15, 20, 25, 30, 35, 40])
  cbar_label.ax.get_yaxis().labelpad = 15
  cbar_label.ax.set_ylabel('Prediction Error (maf)', rotation=270, fontsize = 14)
  cbar_label.ax.set_yticks([-2, -1, 0, 1, 2])
  cbar_label.ax.set_yticklabels(['2 maf\n(Under)','1maf', '0maf', '1maf', '2maf\n(Over)'])
  cbar_label.ax.tick_params(axis='both', which='major', labelsize=12)
  ax_label.set_ylim([3900000, 6100000])
  ax_label.set_xlim([4750000, 35000000])
  ax_label.set_facecolor('gainsboro')
  return ax_label, cbar_label
  
def plot_historical_regression(predictions, r2, independents, dependent, fig_path):
  fig, ax = plt.subplots()
  ax.plot(dependent, independents[:], marker = 'o', markerfacecolor = 'slategray',  markeredgecolor = 'black', markersize = 10, linewidth = 1, color = 'none')
  ax.plot(predictions, independents[:], color = 'black', linewidth = 2.5)
  ax = label_historical_regressions(ax, [r2, ], ['black',], ['Historical Regression',])
  plt.savefig(os.path.join(fig_path, 'nf_unreg_historical_regression_single.png'))
  plt.close()

def plot_historical_regression_multiperiod(predictions_pre1989, predictions_post1989, rsquared_vals, independents, dependent, split_date, fig_path):
  fig, ax = plt.subplots()
  split_idx = split_date - 1963
  color_year = sns.color_palette('RdYlBu', len(dependent))
  for x in range(0, len(dependent)):
    ax.plot([dependent[x],], [independents[x],], color = color_year[x], marker = 'o', markersize = 10, linewidth = 0)
  ax.plot(predictions_pre1989, independents[:split_idx], color = 'indianred', linewidth = 2.5)
  ax.plot(predictions_post1989, independents[split_idx:], color = 'steelblue', linewidth = 2.5)
  ax = label_historical_regressions(ax, rsquared_vals, ['indianred', 'steelblue'], ['Historical Regression, Pre-1989', 'Historical Regression, Post-1989'])
  plt.savefig(os.path.join(fig_path, 'nf_unreg_historical_regression_multi.png'))
  plt.close()
  
def plot_historical_errors(errors, errors_pre1989, errors_post1989, independents, dependent, period = 'single', fig_path=''):
  density = gaussian_kde(errors)
  value_range = np.linspace(-2000000, 2000000, 300)
  total_density = density(value_range)
  if period == 'multi':
    densitya = gaussian_kde(errors_pre1989)
    densityb = gaussian_kde(errors_post1989)
    total_densitya = densitya(value_range)
    total_densityb = densityb(value_range)
    
  fig, ax = plt.subplots()
  ax.fill_between(value_range, total_density, facecolor = 'slategray', edgecolor = 'black', linewidth = 2.5, alpha = 0.7)
  if period == 'multi':
    ax.fill_between(value_range, total_densitya, facecolor = 'indianred', edgecolor = 'black', linewidth = 2.5, alpha = 0.7)
    ax.fill_between(value_range, total_densityb, facecolor = 'steelblue', edgecolor = 'black', linewidth = 2.5, alpha = 0.7)
  
  if period == 'single':
    ax.set_ylim([0, np.max(total_density) * 1.1])
  elif period == 'multi':
    ax.set_ylim([0, max(max(np.max(total_densitya),np.max(total_densityb)), np.max(total_density))  * 1.1])
  
  ax.set_xlim([-2000000, 2000000])
  ax.set_xlabel('Prediction Error, (maf)')
  ax.set_xticks([-2000000, -1000000, 0, 1000000, 2000000])
  ax.set_xticklabels([-2, -1, 0, 1, 2])
  ax.set_yticks([])
  ax.set_yticklabels('')
  plt.savefig(os.path.join(fig_path, 'nf_unreg_historical_errors_' + period + '.png'))
  plt.close()

def plot_demand_nf_regression(natural_flows, tot_demands, unreg_inflows, predictions, rsquared_vals, fig_path):
  fig, ax = plt.subplots()
  color_demand = sns.color_palette('rocket', 5)
  for x in range(0, len(unreg_inflows)):
    color_int = min(max(int((tot_demands[x]/1000000.0 - 4.0) * 2.0), 0), 4)
    ax.plot([natural_flows[x],], [unreg_inflows[x],], color = color_demand[color_int], marker = 'o', markersize = 0.25, linewidth = 0)
  ax.plot(natural_flows, predictions, color = 'black', linewidth = 4.0)
  ax.set_ylabel('Unregulated Powell Inflow')
  ax.set_yticks([5000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000, 40000000])
  ax.set_yticklabels([5, 10, 15, 20, 25, 30, 35, 40])
  ax.set_xlabel('Lees Ferry Natural Flow')
  ax.set_xticks([5000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000, 40000000])
  ax.set_xticklabels([5, 10, 15, 20, 25, 30, 35, 40])
  ax.text(5500000, 30000000, 'R*2: ' + str(int(rsquared_vals[0] * 1000) / 1000.0), fontsize = 14, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  ax.text(5500000, 27500000, 'R*2 < 10maf: ' + str(int(rsquared_vals[1]* 1000) / 1000.0), fontsize = 14, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  ax.text(5500000, 25000000, 'R*2 < 12.5maf: ' + str(int(rsquared_vals[2] * 1000) / 1000.0), fontsize = 14, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  ax.text(5500000, 22500000, 'R*2 > 25maf: ' + str(int(rsquared_vals[3] * 1000) / 1000.0), fontsize = 14, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  custom_lines = []
  legend_labels = []
  for maf_cnt, maf in enumerate([4, 4.5, 5, 5.5, 6]):
    custom_lines.append(Line2D([0], [0], marker = 'o', markerfacecolor = color_demand[maf_cnt], markeredgecolor = 'black', color = 'none', lw = 0))
    legend_labels.append(str(maf) + 'maf')
  custom_lines.append(Line2D([0], [0], color = 'black', lw = 5))
  legend_labels.append('Regression')
  ax.legend(list(reversed(custom_lines)), list(reversed(legend_labels)), loc = 'lower right', prop={'size': 12})
  ax.set_ylim([0, 35000000])
  ax.set_xlim([4750000, 35000000])
  ax.set_facecolor('gainsboro')
  plt.savefig(os.path.join(fig_path, 'nf_unreg_regression.png'))
  plt.close()
  
def plot_demand_nf_errors(natural_flows, tot_demands, unreg_inflows, errors, val_range, fig_path):
  fig, ax = plt.subplots()
  min_val = val_range[0] * 1.0
  range_val = val_range[1] - val_range[0]
  color_demand = sns.color_palette('RdYlBu', 101)
  for x in range(0, len(unreg_inflows)):
    color_int = min(max(int(100 * (errors[x]-min_val) / range_val), 0), 100)
    ax.plot([natural_flows[x],], [tot_demands[x],], color = color_demand[color_int], marker = 'o', markersize = 2.5, linewidth = 0)
  cmap = cm.get_cmap("RdYlBu")
  norm = colors.Normalize(-2, 2)
  cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
  ax, cbar = label_error_plots(ax, cbar)
  plt.savefig(os.path.join(fig_path, 'nf_unreg_errors.png'))
  plt.close()


def plot_demand_nf_regression_multivariate(natural_flows, tot_demands, unreg_inflows, predictions, rsquared_vals, fig_path):
  fig, ax = plt.subplots(2)
  color_demand = sns.color_palette('rocket', 5)
  for x in range(0, len(unreg_inflows)):
    color_int = min(max(int((tot_demands[x]/1000000.0 - 4.0) * 2.0), 0), 4)
    ax[0].plot([natural_flows[x],], [unreg_inflows[x],], color = color_demand[color_int], marker = 'o', markersize = 0.25, linewidth = 0)
    ax[1].plot([natural_flows[x],], [predictions[x],], color = color_demand[color_int], marker = 'o', markersize = 0.25, linewidth = 0)
    
  for ax_use in range(0, 2):
    ax[ax_use].set_yticks([5000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000, 40000000])
    ax[ax_use].set_yticklabels([5, 10, 15, 20, 25, 30, 35, 40])
    ax[ax_use].set_xlim([4750000, 35000000])
    ax[ax_use].set_ylim([0, 35000000])
    ax[ax_use].set_facecolor('gainsboro')
  ax[0].set_ylabel('Unregulated Powell Inflow\nObserved')
  ax[1].set_ylabel('Unregulated Powell Inflow\nPredicted')
  ax[1].text(5500000, 30000000, 'R*2: ' + str(int(rsquared_vals[0] * 1000) / 1000.0), fontsize = 10, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  ax[1].text(5500000, 27500000, 'R*2 < 10maf: ' + str(int(rsquared_vals[1]* 1000) / 1000.0), fontsize = 10, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  ax[1].text(5500000, 25000000, 'R*2 < 12.5maf: ' + str(int(rsquared_vals[2] * 1000) / 1000.0), fontsize = 10, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  ax[1].text(5500000, 22500000, 'R*2 > 25maf: ' + str(int(rsquared_vals[3] * 1000) / 1000.0), fontsize = 10, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  ax[1].set_xlabel('Lees Ferry Natural Flow')
  ax[1].set_xticks([5000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000])
  ax[1].set_xticklabels([5, 10, 15, 20, 25, 30, 35])
  ax[0].set_xticks([])
  ax[0].set_xticklabels('')
  custom_lines = []
  legend_labels = []
  for maf_cnt, maf in enumerate([4, 4.5, 5, 5.5, 6]):
    custom_lines.append(Line2D([0], [0], marker = 'o', markerfacecolor = color_demand[maf_cnt], markeredgecolor = 'black', color = 'none', lw = 0))
    legend_labels.append(str(maf) + 'maf')
  ax[0].legend(list(reversed(custom_lines)), list(reversed(legend_labels)), loc = 'upper left',ncol=2, prop={'size': 10})
  plt.savefig(os.path.join(fig_path, 'nf_unreg_regression_multivariate.png'))
  plt.close()

def plot_demand_nf_errors_multivariate(natural_flows, tot_demands, unreg_inflows, errors, val_range, fig_path):
  fig, ax = plt.subplots()
  color_demand = sns.color_palette('RdYlBu', 101)
  min_val = val_range[0] * 1.0
  range_val = val_range[1] - val_range[0]
  for x in range(0, len(unreg_inflows)):
    color_int = min(max(int(100 * (errors[x]-min_val) / range_val), 0), 100)
    ax.plot([natural_flows[x],], [tot_demands[x],], color = color_demand[color_int], marker = 'o', markersize = 2.5, linewidth = 0)
  cmap = cm.get_cmap("RdYlBu")
  norm = colors.Normalize(-2, 2)
  cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
  ax, cbar = label_error_plots(ax, cbar)
  plt.savefig(os.path.join(fig_path, 'nf_unreg_errors_multivariate.png'))
  plt.close()

def plot_logistic_regression(natural_flows, tot_demands, unreg_inflows, predictions, rsquared_vals, fig_path): 
  fig, ax = plt.subplots(2)
  color_demand = sns.color_palette('rocket', 5)
  for x in range(0, len(unreg_inflows)):
    color_int = min(max(int((tot_demands[x]/1000000.0 - 4.0) * 2.0), 0), 4)
    ax[0].plot([natural_flows[x],], [unreg_inflows[x],], color = color_demand[color_int], marker = 'o', markersize = 0.25, linewidth = 0)
    ax[1].plot([natural_flows[x],], [predictions[x] * 1000000.0,], color = color_demand[color_int], marker = 'o', markersize = 0.25, linewidth = 0)
  for ax_use in range(0, 2):
    ax[ax_use].set_yticks([5000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000, 40000000])
    ax[ax_use].set_yticklabels([5, 10, 15, 20, 25, 30, 35, 40])
    ax[ax_use].set_xlim([4750000, 35000000])
    ax[ax_use].set_ylim([0, 35000000])
    ax[ax_use].set_facecolor('gainsboro')
  ax[0].set_ylabel('Unregulated Powell Inflow\nObserved')
  ax[1].set_ylabel('Unregulated Powell Inflow\nPredicted')
  ax[1].text(5500000, 30000000, 'R*2: ' + str(int(rsquared_vals[0] * 1000) / 1000.0), fontsize = 10, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  ax[1].text(5500000, 27500000, 'R*2 < 10maf: ' + str(int(rsquared_vals[1]* 1000) / 1000.0), fontsize = 10, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  ax[1].text(5500000, 25000000, 'R*2 < 12.5maf: ' + str(int(rsquared_vals[2] * 1000) / 1000.0), fontsize = 10, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  ax[1].text(5500000, 22500000, 'R*2 > 25maf: ' + str(int(rsquared_vals[3] * 1000) / 1000.0), fontsize = 10, horizontalalignment='left', verticalalignment='center', color = 'crimson')
  ax[1].set_xlabel('Lees Ferry Natural Flow')
  ax[1].set_xticks([5000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000])
  ax[1].set_xticklabels([5, 10, 15, 20, 25, 30, 35])
  ax[0].set_xticks([])
  ax[0].set_xticklabels('')
  custom_lines = []
  legend_labels = []
  for maf_cnt, maf in enumerate([4, 4.5, 5, 5.5, 6]):
    custom_lines.append(Line2D([0], [0], marker = 'o', markerfacecolor = color_demand[maf_cnt], markeredgecolor = 'black', color = 'none', lw = 0))
    legend_labels.append(str(maf) + 'maf')
  ax[0].legend(list(reversed(custom_lines)), list(reversed(legend_labels)), loc = 'upper left',ncol=2, prop={'size': 10})
  plt.savefig(os.path.join(fig_path, 'nf_unreg_regression_logistic.png'))
  plt.close()
  
def plot_logistic_error(natural_flows, tot_demands, unreg_inflows, errors, val_range, fig_path): 
  fig, ax = plt.subplots()
  min_val = val_range[0] * 1.0
  range_val = val_range[1] - val_range[0]
  color_demand = sns.color_palette('RdYlBu', 101)
  for x in range(0, len(unreg_inflows)):
    color_int = min(max(int(100 * (errors[x]-min_val) / range_val), 0), 100)
    ax.plot([natural_flows[x],], [tot_demands[x],], color = color_demand[color_int], marker = 'o', markersize = 2.5, linewidth = 0)
  cmap = cm.get_cmap("RdYlBu")
  norm = colors.Normalize(-2, 2)
  cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
  ax, cbar = label_error_plots(ax, cbar)
  plt.savefig(os.path.join(fig_path, 'nf_unreg_errors_logistic.png'))
  plt.close()
