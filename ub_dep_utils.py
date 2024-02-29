import os
import json
import math
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import statsmodels.api as sm


def read_json_file(timestep, label, filename_json):
  # reads json file of historical powell unregulated inflow (from HDB)
  with open(filename_json) as f:
    data = json.load(f)
    timeseries_data = data['data']
    dates = []
    values = []
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
  
  return independents, dependent

def make_historical_regression_single(independents, dependent):
  # fit linear regression over historical period
  # make predictions, calculate R2
  # note: no train/test split
  nf_coef = LinearRegression()
  nf_coef.fit(independents, dependent)
  predictions =  nf_coef.predict(independents)
  errors = np.asarray(predictions) - np.asarray(dependent)
  r2 = r2_score(dependent,predictions)
    
  return predictions, errors, r2 
   
def make_historical_regression_multi(independents, dependent, split_date):   
  # fit linear regressions over 2 historical periods
  # make predictions, calculate R2
  # note: no train/test split
  split_idx = split_date - 1963
  nf_coef_pre1989 = LinearRegression()
  nf_coef_pre1989.fit(independents[:split_idx,:], dependent[:split_idx])

  nf_coef_post1989 = LinearRegression()
  nf_coef_post1989.fit(independents[split_idx:,:], dependent[split_idx:])

  predictions_pre1989 =  nf_coef_pre1989.predict(independents[:split_idx,:])
  predictions_post1989 =  nf_coef_post1989.predict(independents[split_idx:,:])
  errors_pre1989 = np.asarray(predictions_pre1989) - np.asarray(dependent[:split_idx])
  errors_post1989 = np.asarray(predictions_post1989) - np.asarray(dependent[split_idx:])
  r2_pre1989 = r2_score(np.asarray(dependent[:split_idx]),predictions_pre1989)
  r2_post1989 = r2_score(np.asarray(dependent[split_idx:]),predictions_post1989)
    
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
  input_filename_list = ['Computed State Depletions.UB Annual Normal.csv', 'TotVal.Powell.csv', 
                         'Powell.Inflow.csv', 'PowellForecastData.Unreg Inflow no Error.csv']
  demand_scenario_list = ['2016Dems', 'UB90prctDems', 'UB4mafDems', 'UB4.5mafDems', 'UB5mafDems', 'UB5.5mafDems', 'UB6mafDems']
  input_labels = ['demands', 'nf', 'inflow', 'unreg_inflow']
  for lab in input_labels:
    data_lists[lab] = np.zeros((year_range[1] - year_range[0] + 1) * num_traces * len(demand_scenario_list))  
  for dmd_cnt, demand_use in enumerate(demand_scenario_list):
    folder_read = os.path.join('data','robustness_data','policy_HR01,' + demand_use + ',CRMMS_mid_Trace30')
    for file_read, input_key in zip(input_filename_list, input_labels):
      input_df_dict[input_key] = pd.read_csv(os.path.join(folder_read, file_read), index_col = 0)
      input_df_dict[input_key] = input_df_dict[input_key][pd.notna(input_df_dict[input_key]['Trace1'])]
      input_df_dict[input_key].index = pd.to_datetime(input_df_dict[input_key].index)
    input_df_dict['nf'] = input_df_dict['nf'].resample("Y").sum()
    input_df_dict['inflow'] = input_df_dict['inflow'].resample("Y").sum()
    input_df_dict['unreg_inflow'] = input_df_dict['unreg_inflow'].resample("Y").sum()
    for year_use in range(year_range[0], year_range[1] + 1):
      for trace_no in range(0, num_traces):
        datetime_index = datetime.datetime(year_use, 12, 31, 0, 0)
        cnt_idx = trace_no + (year_use - year_range[0]) * num_traces + dmd_cnt * num_traces * (year_range[1] - year_range[0] + 1)
        trace_col = 'Trace' + str(trace_no + 1)
        for lab in input_labels:
          data_lists[lab][cnt_idx] = float(input_df_dict[lab].loc[datetime_index, trace_col]) * 1.0
  natural_flows = np.asarray(natural_flows)
  powell_inflows = np.asarray(powell_inflows)
  tot_demands = np.asarray(tot_demands)
  unreg_inflows = np.asarray(unreg_inflows)
  
  return data_lists['nf'], data_lists['inflow'], data_lists['demands'], data_lists['unreg_inflow']

def fit_regression(ind, dep):
  # fit linear regression and calculate R2 over different
  # parts of the distribution
  nf_coef = LinearRegression()
  nf_coef.fit(ind, dep)
  predictions =  nf_coef.predict(ind)
  errors = np.asarray(nf_coef.predict(ind)) - np.asarray(dep)

  agroup = ind[:,0] < 10000000
  bgroup = ind[:,0] < 12500000
  cgroup = ind[:,0] > 25000000
  r2 = r2_score(dep,predictions)
  r2a = r2_score(dep[agroup], nf_coef.predict(ind[agroup,:]))
  r2b = r2_score(dep[bgroup], nf_coef.predict(ind[bgroup,:]))
  r2c = r2_score(dep[cgroup], nf_coef.predict(ind[cgroup,:]))

  est = sm.OLS(dep, ind)
  est2 = est.fit()
  #print(est2.summary())
  
  return predictions, errors, [r2, r2a, r2b, r2c]

def make_univariate_regression(natural_flows, unreg_inflows):
  # set up training data for univariate regression (w/ constant)
  independents = np.zeros((len(natural_flows),2))
  independents[:,0] = natural_flows
  independents[:,1] = np.ones(len(natural_flows))
  
  return fit_regression(independents, unreg_inflows)

def make_multivariate_regression(natural_flows, tot_demands, unreg_inflows):
  # set up training data for multivariate regression (w/ constant)
  independents = np.zeros((len(natural_flows),3))
  independents[:,0] = natural_flows
  independents[:,1] = tot_demands
  independents[:,2] = np.ones(len(natural_flows))

  return fit_regression(independents, unreg_inflows)
  
def fit_logistic_regression(natural_flows, tot_demands, unreg_inflows): 
  # fit multivariate logistic regression
  def test(X, a, b, c, d, e):
    x, y = X
    final_val = np.zeros(len(x))
    for aa in range(0, len(x)):
      final_val[aa] = x[aa] - a * (1.0 - (1.0/(1 + math.exp(-1*e*(x[aa]-d)))))  - (y[aa]) * (1.0/(1 + math.exp(-1*c*(x[aa]-b))))
    return final_val
  
  p0 = (2.5, 15.0, 0.1, 15.0, 0.1)
  parameters, _ = curve_fit(test, (natural_flows/1000000.0, tot_demands/1000000.0) , unreg_inflows/1000000.0, p0)
  predictions = np.zeros(len(natural_flows))
  errors = np.zeros(len(natural_flows))
  obs_cnt = 0
  for nf_obs, dm_obs in zip(natural_flows, tot_demands):
    nf_input = nf_obs / 1000000.0
    dm_input = dm_obs / 1000000.0
    predictions[obs_cnt] = nf_input - (dm_input/(1 + math.exp(-1*parameters[2]*(nf_input-parameters[1])))) - parameters[0]* (1.0 - (1.0/(1 + math.exp(-1*parameters[4]*(nf_input-parameters[3]))))) 
    errors[obs_cnt] = predictions[obs_cnt]*1000000.0 - unreg_inflows[obs_cnt]
    obs_cnt += 1

  agroup = natural_flows < 10000000
  bgroup = natural_flows < 12500000
  cgroup = natural_flows > 25000000
  r2 = r2_score(unreg_inflows,predictions * 1000000)
  r2a = r2_score(unreg_inflows[agroup], predictions[agroup] * 1000000)
  r2b = r2_score(unreg_inflows[bgroup], predictions[bgroup] * 1000000)
  r2c = r2_score(unreg_inflows[cgroup], predictions[cgroup] * 1000000)

  return predictions, errors, [r2, r2a, r2b, r2c]
