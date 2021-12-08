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

# globals --
testsize = 0.2
ignore_ = False


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
# Poly
# polyReg(data.iloc[0:100], target.iloc[0:100], 5)

if not ignore_:
    # Dummy
    evm = varyFeatureNumber(
        data.iloc[:50], target.iloc[:50], 'dummy', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[Dummy] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    # Linear Regression
    evm = varyFeatureNumber(
        data.iloc[:25], target.iloc[:25], 'linreg', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[LR] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    # Lasso
    evm = varyFeatureNumber(
        data.iloc[:25], target.iloc[:25], 'lasso', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[Lasso] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    # Decision Tree
    evm = varyFeatureNumber(data, target, 'tree', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[DTR] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    # K-Neighbors
    evm = varyFeatureNumber(data, target, 'kn', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[KN] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    # Random Forest
    evm = varyFeatureNumber(data, target, 'forest', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[RFR] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    exit()

    # Poly
    evm = varyFeatureNumber(data, target, 'poly', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[Poly] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    """ # Ridge
    evm = varyFeatureNumber(data, target, 'ridge', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[Ridge] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    ) """

    # @TODO - Multi-layer Perceptron
    evm = varyFeatureNumber(data, target, 'mlp', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[MLP] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )


# maus resultados :(
# oqeq pode estar a acontecer?
# - as features selecionadas não são indicativas do numero de lesoes
#       * experimentar outro metodo de feature selection:
#         https://machinelearningmastery.com/calculate-feature-importance-with-python/
# - o dataset que eu fiz está mal construido

print('Done')
