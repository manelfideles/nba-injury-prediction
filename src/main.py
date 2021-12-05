"""
This is the main file of
the predictor. All the training
and results are here.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

from dependencies import *
from utils import *
from tests import *

# Dataset EDA
mainDataset = importData(processedDataDir, 'mainDataset.csv')
# getStatsAllSeasons(mainDataset)

data, target = mainDataset.drop(columns={
    'Player',
    'Team',
    'Season',
    '# of Injuries (Season)'}), mainDataset['# of Injuries (Season)']

# No-Injury predictor
test_0inj = pd.DataFrame(0, index=np.arange(
    len(target)), columns=['# of Injuries (Season)'])

# Feature Selection using Pearson's corrcoeff
train, test = varyFeatureNumber(data, target, 'linreg', 0.4)

# Linear Regression
plotMultipleLineGraphs(
    train,
    title='[LR] # of Features vs MSE, MAE, RMSE, R^2 - Training set'
)
plotMultipleLineGraphs(
    test,
    title='[LR] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
)

# Decision Tree
train, test = varyFeatureNumber(data, target, 'tree', 0.4)
plotMultipleLineGraphs(
    train,
    title='[DTR] # of Features vs MSE, MAE, RMSE, R^2 - Training set'
)
plotMultipleLineGraphs(
    test,
    title='[DTR] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
)

# Random Forest
train, test = varyFeatureNumber(data, target, 'forest', 0.4)
plotMultipleLineGraphs(
    train,
    title='[RFR] # of Features vs MSE, MAE, RMSE, R^2 - Training set'
)
plotMultipleLineGraphs(
    test,
    title='[RFR] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
)

print('Done')
