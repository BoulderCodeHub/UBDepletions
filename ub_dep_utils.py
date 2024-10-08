import os
import json
import math
import pandas as pd
import numpy as np
import datetime
from scipy.optimize import curve_fit
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

def read_json_file(timestep, label, filename_json):
  # reads json file of historical powell unregulated inflow (from HDB)
  with open(filename_json) as f:
    data = json.load(f)
    timeseries_data = data['data']
    dates = []
    values = []
#    for x in range(0, 12):
#      values_monthly[str(x+1)] = []
    start_record = False
    for obs in timeseries_data:
      if obs['value']:
        date_read = obs['datetime'].split('-')
        if int(date_read[1]) == 10:
          start_record = True      
          annual_val = 0.0
        if start_record:
          annual_val += float(obs['value'])          
          if int(date_read[1]) == 9 or timestep == 'annual':
            values.append(annual_val)
            dates.append(int(date_read[0]))
      else:
        print(obs['datetime'])
        
  return pd.DataFrame(values, index = dates, columns = [label, ])
  
def read_natural_flow_obs(timestep, label, filename_csv):
  # reads csv of full-natural-flow for lees ferry (from CRSS inputs)
  historical_lf = pd.read_csv(filename_csv, index_col = 0)
  historical_lf.index = pd.to_datetime(historical_lf.index)
  start_record = False
  dates = []
  values = []
#  for x in range(0, 12):
#    values_monthly[str(x+1)] = []
  for index, row in historical_lf.iterrows():
    if index.month == 10:
      start_record = True
      annual_val = 0.0
    if start_record:
      annual_val += float(row['HistoricalNaturalFlow.AboveLeesFerry'])
      if index.month == 9:
        values.append(annual_val)
        dates.append(index.year)

  return pd.DataFrame(values, index = dates, columns = [label, ])

def regression_natural_flows(historical_unreg, historical_nf):
  # clips historical natural flow/unregulated inflow data to 
  # ensure they cover the same time period
  independent = np.asarray(historical_unreg)
  dependent = np.asarray(historical_nf)
  if historical_unreg.index[0] > historical_nf.index[0]:
    dependent = dependent[(historical_unreg.index[0] - historical_nf.index[0]):]
  elif historical_nf.index[0] > historical_unreg.index[0]:
    independent = independent[(historical_nf.index[0] - historical_unreg.index[0]):]
  if historical_unreg.index[-1] > historical_nf.index[-1]:
    independent = independent[:-1*(historical_unreg.index[-1] - historical_nf.index[-1])]
  elif historical_nf.index[-1] > historical_unreg.index[-1]:
    dependent = dependent[:-1*(historical_nf.index[-1] - historical_unreg.index[-1])]

  independents = np.zeros((len(independent), 2))
  independents[:,0] = independent
  independents[:,1] = np.ones(len(independent))
  
  return independent, dependent

def make_historical_regression_single(independents, dependent):
  # fit linear regression over historical period
  # make predictions, calculate R2
  # note: no train/test split
  slope, intercept, r_value, p_value, std_err = stats.linregress(independents, dependent)
  predictions = intercept + slope * independents
  errors = np.asarray(predictions) - np.asarray(dependent)
  r2 = np.power(r_value, 2)
    
  return predictions, errors, r2 
   
def make_historical_regression_multi(independents, dependent, split_date):   
  # fit linear regressions over 2 historical periods
  # make predictions, calculate R2
  # note: no train/test split
  split_idx = split_date - 1963
  slope, intercept, r_value, p_value, std_err = stats.linregress(independents[:split_idx], dependent[:split_idx])
  predictions_pre1989 = intercept + slope * independents[:split_idx]
  r2_pre1989 = np.power(r_value,2)

  slope, intercept, r_value, p_value, std_err = stats.linregress(independents[split_idx:], dependent[split_idx:])
  predictions_post1989 = intercept + slope * independents[split_idx:]
  r2_post1989 = np.power(r_value,2)

  errors_pre1989 = np.asarray(predictions_pre1989) - np.asarray(dependent[:split_idx])
  errors_post1989 = np.asarray(predictions_post1989) - np.asarray(dependent[split_idx:])
    
  return predictions_pre1989, predictions_post1989, errors_pre1989, errors_post1989, [r2_pre1989, r2_post1989]

def make_regression_inputs(year_range, num_traces):
  # read simulated data from all demand scenarios
  # need natural flow, ub demands, and unregulated inflow
  # actual powell inflows are calculated - can be used in place of unreg
  natural_flows = []
  powell_inflows = []
  tot_demands = []
  unreg_inflows = []
  input_df_dict = {}
  data_lists = {}
  data_lists_monthly = {}
  input_filename_list = ['Computed State Depletions.UB Annual Normal.csv', 'TotVal.Powell.csv', 
                         'Powell.Inflow.csv', 'PowellForecastData.Unreg Inflow no Error.csv']
  #demand_scenario_list = ['2016Dems', 'UB90prctDems', 'UB4mafDems', 'UB4.5mafDems', 'UB5mafDems', 'UB5.5mafDems', 'UB6mafDems']
  demand_scenario_list = ['UB5.5mafDems', ]
  input_labels = ['demands', 'nf', 'inflow', 'unreg_inflow']
  for lab in input_labels:
    data_lists[lab] = np.zeros((year_range[1] - year_range[0] + 1) * num_traces * len(demand_scenario_list))  
    data_lists_monthly[lab] = np.zeros(((year_range[1] - year_range[0] + 1) * num_traces * len(demand_scenario_list), 12))  
  for dmd_cnt, demand_use in enumerate(demand_scenario_list):
    folder_read = os.path.join('data','robustness_data','policy_HR01,' + demand_use + ',CRMMS_mid_Trace30')
    for file_read, input_key in zip(input_filename_list, input_labels):
      input_df_dict[input_key] = pd.read_csv(os.path.join(folder_read, file_read), index_col = 0)
      input_df_dict[input_key] = input_df_dict[input_key][pd.notna(input_df_dict[input_key]['Trace1'])]
      input_df_dict[input_key].index = pd.to_datetime(input_df_dict[input_key].index)
    input_df_dict_annual = {}
    input_df_dict_annual['nf'] = input_df_dict['nf'].resample("Y").sum()
    input_df_dict_annual['inflow'] = input_df_dict['inflow'].resample("Y").sum()
    input_df_dict_annual['unreg_inflow'] = input_df_dict['unreg_inflow'].resample("Y").sum()
    input_df_dict_annual['demands'] = input_df_dict['demands'].resample("Y").sum()
    for year_use in range(year_range[0], year_range[1] + 1):
      for trace_no in range(0, num_traces):
        datetime_index = datetime.datetime(year_use, 12, 31, 0, 0)
        cnt_idx = trace_no + (year_use - year_range[0]) * num_traces + dmd_cnt * num_traces * (year_range[1] - year_range[0] + 1)
        trace_col = 'Trace' + str(trace_no + 1)
        for lab in input_labels:
          data_lists[lab][cnt_idx] = float(input_df_dict_annual[lab].loc[datetime_index, trace_col]) * 1.0
    for lab in input_labels:
      for index, row in input_df_dict[lab].iterrows():
        if index > datetime.datetime(year_range[0], 1, 1, 0, 0) and index < datetime.datetime(year_range[1]+1, 1, 1, 0, 0):    
          for trace_no in range(0, num_traces):
            trace_col = 'Trace' + str(trace_no + 1)
            cnt_idx = trace_no + (index.year - year_range[0]) * num_traces + dmd_cnt * num_traces * (year_range[1] - year_range[0] + 1)
            if lab == 'demands':
              for mNo in range(0, 12):
                data_lists_monthly[lab][cnt_idx, mNo] = float(input_df_dict[lab].loc[index, trace_col]) * 1.0
            else:
              data_lists_monthly[lab][cnt_idx, index.month - 1] = float(input_df_dict[lab].loc[index, trace_col]) * 1.0
      
      
  
  return data_lists['nf'], data_lists['inflow'], data_lists['demands'], data_lists['unreg_inflow'], data_lists_monthly['nf'], data_lists_monthly['inflow'], data_lists_monthly['demands'], data_lists_monthly['unreg_inflow']

def fit_regression(ind, dep):
  # fit linear regression and calculate R2 over different
  # parts of the distribution
  slope, intercept, r_value, p_value, std_err = stats.linregress(ind, dep)
  predictions = intercept + slope * ind
  errors = np.asarray(predictions) - np.asarray(dep)
  r2 = np.power(r_value, 2)
  
  return predictions, errors, [r2, r2, r2, r2]

def make_univariate_regression(natural_flows, unreg_inflows):
  # set up training data for univariate regression (w/ constant)
  #independents = np.zeros((len(natural_flows),2))
  #independents[:,0] = natural_flows
  #independents[:,1] = np.ones(len(natural_flows))
  
  return fit_regression(natural_flows, unreg_inflows)

def make_multivariate_regression(natural_flows, tot_demands, unreg_inflows):
  # set up training data for multivariate regression (w/ constant)
  independents = np.zeros((len(natural_flows),3))
  independents[:,0] = natural_flows
  independents[:,1] = tot_demands
  independents[:,2] = np.ones(len(natural_flows))
  est = sm.OLS(unreg_inflows, independents).fit() 

  return fit_regression(natural_flows, unreg_inflows)
  
def fit_logistic_regression(natural_flows, tot_demands, unreg_inflows): 
  # fit multivariate logistic regression
  def test(X, a, b, c, d, e, f):
    x, y, z = X
    final_val = np.zeros(len(x))
    for aa in range(0, len(x)):
      final_val[aa] = x[aa] - a * (1.0 - (1.0/(1 + math.exp(-1*e*(x[aa]-d)))))  - f * (y[aa]/(1 + math.exp(-1*c*(x[aa]-b))))
    return final_val
  
  p0 = (0.1, 0.1, 10.0, 0.1, 10.0, 1.0)
  param_bounds=([0,-np.inf, -np.inf, -np.inf, -np.inf, 0],[np.max(natural_flows)/1000000,np.inf,np.inf,np.inf,np.inf, 1.0])
  parameters, _ = curve_fit(test, (natural_flows/1000000.0, tot_demands/1000000.0 ,unreg_inflows/1000000.0), unreg_inflows/1000000.0, bounds = param_bounds, maxfev = 5000)
  
  predictions = np.zeros(len(natural_flows))
  errors = np.zeros(len(natural_flows))
  obs_cnt = 0
  for nf_obs, dm_obs in zip(natural_flows, tot_demands):
    nf_input = nf_obs / 1000000.0
    dm_input = dm_obs / 1000000.0
    predictions[obs_cnt] = nf_input - (parameters[5] * dm_input/(1 + math.exp(-1*parameters[2]*(nf_input-parameters[1])))) - parameters[0]* (1.0 - (1.0/(1 + math.exp(-1*parameters[4]*(nf_input-parameters[3]))))) 
    errors[obs_cnt] = predictions[obs_cnt]*1000000.0 - unreg_inflows[obs_cnt]
    obs_cnt += 1

  agroup = natural_flows < 10000000
  bgroup = natural_flows < 12500000
  cgroup = natural_flows > 25000000
  mean_val = np.mean(unreg_inflows)
  sum_errors = 0.0
  sum_tot = 0.0
  for xxx in range(0, len(unreg_inflows)):
    sum_errors += np.power(unreg_inflows[xxx] - predictions[xxx]*1000000, 2)
    sum_tot += np.power(unreg_inflows[xxx] - mean_val, 2)
  r2 = 1.0 - sum_errors/sum_tot
     

  return predictions, errors, [r2, r2, r2, r2], parameters
