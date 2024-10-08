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
this script creates parameter values for the estimation of Upper Basin Depletions
from Natural Flow and Upper Basin Full Demand Schedules that follow the form:
UB Depletions =  Natural Flow - param1 * (1.0 - (1.0/(1 + e ^ (-1 * param3*( Natural Flow - param2)))))  - param4 * (Upper Basin Full Demand/(1 + e ^ (-1 * param6* (Natural Flow- param5))))

Parameters are written to the file 'monthly_params.csv' - 12 rows and 6 columns
the first row corresponds to January
each row in monthly_params.csv contain 6 parameters, calibrated for individual months
these parameters are copied into the table slot UBDepletions.LogisticParameters (also 12 x 6)

param1 is labeled Magnitude1 (first column)
param2 is labeled Location1 (second column)
param3 is labeled Scale1 (third column)
param4 is labeled Magnitude2 (fourth column)
param5 is labeled Location2 (fifth column)
param6 is labeled Scale2 (sixth column)

the table slot UbDepletions.LogisticParameters is used to
evaluate expression slot 'UBDepletions.UBDepletionsEstimate'
at the beginning of a CRSS run, this estimate is used within simulations for forecasting

The parameter file can be generated with the command below

```bash
# for single run
python -W ignore run_regression_analysis.py


