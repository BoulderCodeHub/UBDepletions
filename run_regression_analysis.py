import os
import ub_dep_utils as ubdutil
import ub_dep_plots as ubdplt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# this script uses historical & simulated data on
# lees ferry natural flows and upper basin demands
# to estimate unregulated inflows into powell
# code calculates in 4 steps:
# 1. historical regression (1963 - 2020 and 1963 - 1988/ 1989-2020)
# 2. linear regression w/ simulated data
# 3. multivariate regression w/ simulated data
# 4. multivariate logistic regression w/ simulated data
# 5. UB demand scenarios are taken from the 6 web tool scenarios

historical_split_date = 1989 # year of UB depeletion step change in historical record
simulation_year_range = [2027, 2056] # range of simulation data w/ different UB demand scenarios
simulation_traces = 400 # number of hydrology traces in simulation data
error_range = [-2000000, 2000000] # plotting range for prediction errors

# Calculate Historical regressions
print('read historical data')
filename_hdb = 'data/historical_powell_unregulated_inflow.json'
filename_csv = 'data/historical_lees_ferry_flow.csv'
historical_unreg = ubdutil.read_json_file('monthly', 'unreg', filename_hdb)
historical_unreg.to_csv('annual_historical_unregulated_inflow.csv')
historical_nf = ubdutil.read_natural_flow_obs('monthly', 'nf', filename_csv)

print('make historical regressions')
independents, dependent = ubdutil.regression_natural_flows(historical_unreg['unreg'], historical_nf['nf'])
predictions, errors, r2 = ubdutil.make_historical_regression_single(independents, dependent)
predictions_pre1989, predictions_post1989, errors_pre1989, errors_post1989, rsquared_vals = ubdutil.make_historical_regression_multi(independents, dependent, historical_split_date)

# Plot historical regressions
print('plot historical figures')
if not os.path.isdir('figures'):
  os.mkdir('figures')
ubdplt.plot_historical_regression(predictions, r2, independents, dependent)
ubdplt.plot_historical_regression_multiperiod(predictions_pre1989, predictions_post1989, rsquared_vals, independents, dependent, historical_split_date)
ubdplt.plot_historical_errors(errors, errors_pre1989, errors_post1989, independents, dependent)
ubdplt.plot_historical_errors(errors, errors_pre1989, errors_post1989, independents, dependent, period = 'multi')

# Calculate regressions w/ simulated data
# read simulated data
print('fit regression models')
natural_flows, powell_inflows, tot_demands, unreg_inflows, natural_flows_monthly, powell_inflows_monthly, tot_demands_monthly, unreg_inflows_monthly = ubdutil.make_regression_inputs(simulation_year_range, simulation_traces)
# univariate linear regression
print('....univariate')
predictions_uv, errors_uv, rsquared_vals_uv = ubdutil.make_univariate_regression(natural_flows, unreg_inflows)
# multivariate linear regression
print('....multivariate')
predictions_mv, errors_mv, rsquared_vals_mv = ubdutil.make_multivariate_regression(natural_flows, tot_demands, unreg_inflows)
# multivariate logistic regression
print('....logistic')
param_table = pd.DataFrame()
for monthNum in range(0, 12):
  predictions_lr, errors_lr, rsquared_vals_lr, params = ubdutil.fit_logistic_regression(natural_flows_monthly[:,monthNum], tot_demands_monthly[:,monthNum], unreg_inflows_monthly[:,monthNum])
  print(monthNum)
  print(params)
  param_table.loc[monthNum, 'Magnitude1'] = float(params[0])
  param_table.loc[monthNum, 'Location1'] = float(params[3])
  param_table.loc[monthNum, 'Scale1'] = float(params[4])
  param_table.loc[monthNum, 'Magnitude2'] = float(params[5])
  param_table.loc[monthNum, 'Location2'] = float(params[1])
  param_table.loc[monthNum, 'Scale2'] = float(params[2])
param_table.to_csv('monthly_params.csv')
  
# Plot linear regression
print('plot linear regression')
ubdplt.plot_demand_nf_regression(natural_flows, tot_demands, unreg_inflows, predictions_uv, rsquared_vals_uv)
ubdplt.plot_demand_nf_errors(natural_flows, tot_demands, unreg_inflows, errors_uv, error_range)

# Plot multivariate linear regression
print('plot multivar regression')
ubdplt.plot_demand_nf_regression_multivariate(natural_flows, tot_demands, unreg_inflows, predictions_mv, rsquared_vals_mv)
ubdplt.plot_demand_nf_errors_multivariate(natural_flows, tot_demands, unreg_inflows, errors_mv, error_range)

# Plot multivariate logistic regression
print('plot logistic regression')
ubdplt.plot_logistic_regression(natural_flows, tot_demands, unreg_inflows, predictions_lr, rsquared_vals_lr)
ubdplt.plot_logistic_error(natural_flows, tot_demands, unreg_inflows, errors_lr, error_range)
