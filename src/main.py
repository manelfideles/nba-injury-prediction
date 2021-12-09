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
reg = False
classif = True
debug = True


# Regression yields bad results (< 20% Accuracy)
# There's a few reasons as to why this might be happening:
# 1) The feature selection algorithm is univariate (Pearson's
# Correlation Coefficient), so it doesn't account for
# 2) The dataset was poorly built and/or has irrelevant features
# 3) A linear relationship doesn't describe the problem and its
# complexities well enough.
if reg:
    # Import data and perform EDA on it
    mainDataset = importData(processedDataDir, 'mainDataset.csv')
    getStatisticalMetrics(mainDataset)

    data, target = mainDataset.drop(columns={
        'Player',
        'Team',
        'Season',
        '# of Injuries (Season)'}), mainDataset['# of Injuries (Season)']

    modelnames = [
        'dummy', 'linreg', 'lasso',
        'tree', 'kn', 'forest'
    ]

    for mn in modelnames:
        evm = varyFeatureNumberReg(
            data, target, mn, testsize)
        plotMultiple(
            evm,
            graphtype='line',
            title=f'[{mn.capitalize()}] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
        )

# --
if classif:
    # Import windowed feature vectors (windowsize=3)
    fvs = importData(processedDataDir, 'fvs.csv').drop(
        columns={
            'Player', 'Season', 'Month_2',
            'Age_2', 'Month_3', 'Age_3'
        }
    )

    # windowsize=3 yields 307 observations that are
    # going to be removed.
    # No need to replace them by
    # interpolating, as they constitute only ~3%
    # of the number of total observations in the dataset.
    # So we remove them:
    nan_obs = fvs.shape[0] - fvs.dropna().shape[0]
    print(f'observations < "windowsize": {nan_obs}')
    print(f'Percentage of observations to discard: {nan_obs/fvs.shape[0]}')
    fvs = fvs.dropna().reset_index(drop=True)
    if debug:
        print(fvs.shape)

    # @TODO correr again pq já não me lembro oqeq este código faz
    # significantly imbalanced data (aka #non-injured >>> #injured)
    # statMetrics = getStatisticalMetrics(fvs)
    # plotDistribution(fvs['Injured_1'], statMetrics['Injured_1'])
    # plotDistribution(fvs['Injured_2'], statMetrics['Injured_2'])
    # plotDistribution(fvs['Injured_3'], statMetrics['Injured_3'])

    # Dataset normalization
    data = fvs.iloc[:, :-1].apply(lambda x: (x-x.mean()) / x.std(), axis=1)
    target = fvs.iloc[:, -1]
    print(data.shape)
    print(target.shape)

    # Plot class imbalance and univariate correlation
    imbalance = target.value_counts()
    if debug:
        print(imbalance)
    _, ax = plt.subplots()
    imbalance.plot(kind='barh')
    plt.show()
    plotHeatmap(fvs)

    # Perform PCA analysis with 95% EVR
    evr = 95
    evratios = getEvrs(data)
    pcs = findPCs(evratios, evr + 1)
    plotEvrPc(evratios, pcs)

    X_train, X_test, y_train, y_test = ttSplit(data, target, 0.2)
    clf = make_pipeline(
        ReliefF(n_features_to_select=25, n_neighbors=15),
        RandomForestClassifier(n_jobs=-1)
    )
    print(np.mean(cross_val_score(clf, X_train.to_numpy(), y_train.to_numpy())))

    evms, modelnames = [], [
        'dummy', 'ridge', 'tree',
        'forest', 'kn', 'svm',
        'nb', 'mlp'
    ]
    for mn in modelnames:
        evms += [varyFeatureNumberClassif(data, target, mn, 0.3)]

    for i in range(len(evms)):
        plotResults(evms[i], title=modelnames[i])


print('Done')
