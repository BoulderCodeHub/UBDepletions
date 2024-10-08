# Natural Flow / Unregulated Inflow Regressions

These scripts generate figures detailing regression model performance  
in estimating unregulated inflows into Lake Powell given Lees Ferry  
Natural Flows and Upper Basin pre-shorted demands. Three different  
regression models (univariate linear, multivariate linear, multivariate logistic)  
are tested across 6 different Upper Basin Demand schedules and 400 30-year  
Lees Ferry Natural Flow traces

## Environment setup

The following libraries are required for the project

- [> python3.7](https://www.python.org/downloads/)
- [pandas]
- [numpy]
- [datetime]
- [sklearn]
- [scipy]
- [statsmodels]
- [matplotlib]
- [seaborn]

## Usage
this script creates figures files and 'monthly_params.csv'
generate parameter file with the command below
each row in monthly_params.csv are the 6 parameters for each month
that are read into UBDepletions.LogisticParameters (0 = January)
these parameters are used to calibrate a function of the form
UB Depletions =  Natural Flow - param1 * (1.0 - (1.0/(1 + e ^ (-1 * param3*( Natural Flow - param2)))))  - param4 * (Upper Basin Full Demand/(1 + e ^ (-1 * param6* (Natural Flow- param5))))
this equation is usedto calibrate the monthly UB depletions slot 'UBDepletions.UBDepletionsEstimate'

```bash
# for single run
python -W ignore run_regression_analysis.py


