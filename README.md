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

```bash
# for single run
python -W ignore run_regression_analysis.py

