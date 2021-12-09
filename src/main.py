"""
This is the main file of
the predictor. All the training
and results are here.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

from sklearn.pipeline import make_pipeline
from dependencies import *
from utils import *
from tests import *

# globals --
testsize = 0.2
reg = False

# Regression
"""
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

if reg:
    # Dummy - baseline
    evm = varyFeatureNumberReg(
        data.iloc[:50], target.iloc[:50], 'dummy', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[Dummy] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    # Linear Regression
    evm = varyFeatureNumberReg(
        data.iloc[:25], target.iloc[:25], 'linreg', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[LR] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    # Lasso
    evm = varyFeatureNumberReg(
        data.iloc[:25], target.iloc[:25], 'lasso', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[Lasso] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    # Decision Tree
    evm = varyFeatureNumberReg(data, target, 'tree', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[DTR] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    # K-Neighbors
    evm = varyFeatureNumberReg(data, target, 'kn', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[KN] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )

    # Random Forest
    evm = varyFeatureNumberReg(data, target, 'forest', testsize)
    plotMultiple(
        evm,
        graphtype='line',
        title='[RFR] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
    )


# maus resultados :(
# oqeq pode estar a acontecer?
# - as features selecionadas não são indicativas do numero de lesoes
#       * experimentar outro metodo de feature selection:
#         https://machinelearningmastery.com/calculate-feature-importance-with-python/
# - o dataset que eu fiz está mal construido
# - o problema é demasiado complexo para ser ajustado a uma relação linear
"""

# window = 3
fvs = importData(processedDataDir, 'fvs.csv').drop(
    columns={'Player', 'Season', 'Month_2', 'Age_2', 'Month_3', 'Age_3'})

# number of player with observations < 'windowsize'
# print(f'observations < "windowsize": {fvs.shape[0] - fvs.dropna().shape[0]}')
# plotHeatmap(fvs)

# eliminate rows with nan - @TODO interpolate instead of removing
fvs = fvs.dropna().reset_index()

# significantly imbalanced data (aka #non-injured >>> #injured)
# statMetrics = getStatisticalMetrics(fvs)
# plotDistribution(fvs['Injured_1'], statMetrics['Injured_1'])
# plotDistribution(fvs['Injured_2'], statMetrics['Injured_2'])
# plotDistribution(fvs['Injured_3'], statMetrics['Injured_3'])

# features = fvs.iloc[:, :-1].drop(columns={'Injured_1', 'Injured_2'})
# target = fvs.iloc[-1]

data = fvs.iloc[:, :-1].apply(lambda x: (x-x.mean()) / x.std())
print(data.shape)
target = fvs.iloc[:, -1]
print(target.shape)

# class imbalance
_, ax = plt.subplots()
target.value_counts().plot(kind='barh')
plt.show()

evr = 95  # desired explained variance ratio
evratios = getEvrs(data)
pcs = findPCs(evratios, evr + 1)
plotEvrPc(evratios, pcs)

# Baseline
evms = []
"""
X_train, X_test, y_train, y_test = ttSplit(data, target, 0.2)
clf = make_pipeline(
    ReliefF(n_features_to_select=25, n_neighbors=10),
    RandomForestClassifier(n_jobs=-1)
)
print(np.mean(cross_val_score(clf, X_train.to_numpy(), y_train.to_numpy())))
"""
#
modelnames = ['dummy', 'ridge', 'tree', 'forest', 'kn', 'svm', 'nb', 'mlp']
for mn in modelnames:
    evms += [varyFeatureNumberClassif(data, target, mn, 0.3)]

for i in range(len(evms)):
    plotResults(evms[i], title=modelnames[i])


print('Done')
