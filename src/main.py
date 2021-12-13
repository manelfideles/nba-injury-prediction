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
debug = False
info = False
pca = True
global_testsize = 0.15


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
            data, target, mn, global_testsize)
        plotMultiple(
            evm,
            graphtype='line',
            title=f'[{mn.capitalize()}] # of Features vs MSE, MAE, RMSE, R^2 - Testing set'
        )

# --
if classif:
    # Import windowed feature vectors (windowsize=3)
    fvs = importData(processedDataDir, 'fvs.csv').drop(
        columns={'Month_2', 'Month_3', 'Age_3'}  # {'Player', 'Season'}
    ).rename(columns={'Age_2': 'Age'})

    # windowsize=3 yields 307 observations that are
    # going to be removed.
    # No need to replace them by
    # interpolating, as they constitute only ~3%
    # of the number of total observations in the dataset.
    # So we remove them:
    nan_obs = fvs.shape[0] - fvs.dropna().shape[0]
    if info:
        print(f'Number of observations smaller than windowsize: {nan_obs}')
        print(
            f'Percentage of observations to discard: {round(nan_obs/fvs.shape[0], 3) * 100} %')
    fvs = fvs.dropna().reset_index(drop=True)
    if info:
        print(f'Dataset shape after NaN removal: {fvs.shape}')

    # @TODO correr again pq já não me lembro oqeq este código faz
    # significantly imbalanced data (aka #non-injured >>> #injured)
    # statMetrics = getStatisticalMetrics(fvs)
    # plotDistribution(fvs['Injured_1'], statMetrics['Injured_1'])
    # plotDistribution(fvs['Injured_2'], statMetrics['Injured_2'])
    # plotDistribution(fvs['Injured_3'], statMetrics['Injured_3'])

    # Dataset normalization
    for col in fvs.columns.tolist()[2:-1]:
        if col not in ['Injured_1', 'Injured_2']:
            fvs[col] = (fvs[col] - fvs[col].mean()) / fvs[col].std()
    data = fvs.iloc[:, 2:-1]
    target = fvs.iloc[:, -1]
    if info:
        print(data)
        print(target)
        print(f'Data shape after normalization: {data.shape}')
        print(f'Target shape: {target.shape}')

    # Plot class imbalance and feature correlation
    imbalance = target.value_counts()
    if info:
        print(
            f'Positive (Injured) class examples (#, %): {imbalance.iloc[1]}, {round(imbalance.iloc[1]/target.shape[0], 4) * 100}%')
        imbalance.plot(kind='barh')
        plt.show()
        plotHeatmap(fvs)

    # Perform PCA analysis on the dataset in order to
    # remove the features with residual EVR.
    # For ReliefF: The major drawback of ReliefF
    # is that it does not consider feature dependencies
    # and therefore does not help remove redundant features.
    # PCA combats that issue, by reducing dimensionality through
    # the elimination of features with residual EVR.
    # For SFS (Sequential Forward Selection):
    # PCA is used for dimensionality reduction, as SFS can be
    # very computationally heavy, even with few features to process.
    # @TODO - Perform PCA analysis with 99% EVR
    if info:
        evr = 99  # corresponds to 44 PCs
        evratios = getEvrs(data)
        pcs = findPCs(evratios, evr)
        plotEvrPc(evratios, pcs)

    n_features = 44
    pca_model, data_pca = doPca(data, n_features)

    # Save most important features - returns 30 cols
    mostimportantcols = savePCAFeatures(
        pca_model,
        data.columns.tolist(),
        n_features
    )
    data_pca = data[mostimportantcols]

    # Perform SFS with subset of PCA data
    # to save on time and memory.
    # Subset size is set at 20%.
    # Loop through increasing number of features
    # to measure performance of Sequential Forward Selection.

    samples = np.random.choice(
        data_pca.shape[0],
        size=math.floor(data_pca.shape[0] * 0.1),
        replace=False
    )

    data_pca_subset = data_pca.iloc[samples].reset_index(drop=True)
    target_subset = target.iloc[samples].reset_index(drop=True)

    if info:
        print(data_pca_subset.shape)
        print(target_subset.shape)

    if pca:
        sfs_metrics_pca, sfs_metrics_pca_sfs = [], []

        # Random splitting
        X_train, X_test, y_train, y_test = ttSplit(
            data_pca_subset,
            target_subset,
            global_testsize
        )

        print('Performing prediction w/ SFSelection + PCA...')
        for i in range(1, data_pca.shape[1]):
            print(f'# of Features: {i}')
            clf = make_pipeline(
                SequentialFeatureSelector(
                    KNeighborsClassifier(n_jobs=-1),
                    n_features_to_select=i
                ),
                KNeighborsClassifier(n_jobs=-1)
            ).fit(X_train, y_train)

            # Measuring
            sfs_metrics_pca_sfs += [
                round(roc_auc_score(
                    y_test,
                    clf.predict(X_test),
                    average='weighted'
                ), 4)
            ]

        print('Performing prediction w/ PCA-only...')
        for i in range(1, data_pca.shape[1]):
            print(f'# of Features: {i}')
            clf = KNeighborsClassifier(
                n_jobs=-1).fit(X_train, y_train)

            # Measuring
            sfs_metrics_pca += [
                round(roc_auc_score(
                    y_test,
                    clf.predict(X_test),
                    average='weighted'
                ), 4)
            ]

        _, ax = plt.subplots()
        ax.plot(sfs_metrics_pca_sfs, 'o-', sfs_metrics_pca, 'x-')
        plt.title('[KN] ROC-AUC - PCA+SFS vs PCA-Only')
        plt.show()

    exit()

    # TODO - Feed feature selection output into models
    # Compare models based on the following metrics:
    # Confusion Matrix, Recall, Precision
    # Accuracy, Balanced Accuracy, F1
    # ROC AUC, Precision-Recall AUC
    # while varying the selected features number.

    evms = []
    modelnames = [
        'no-injury', 'dummy', 'ridge',
        'tree', 'forest', 'kn',
        'svm', 'nb', 'mlp'
    ]
    for mn in modelnames:
        evms += [varyFeatureNumberClassif(
            data_pca, target, mn,
            global_testsize,
            balanced=True,
            playercol=fvs['Player'],
            seasoncol=fvs['Season']
        )]

    # Plot evaluation metrics vs number
    # of selected features
    for i in range(len(evms)):
        plotResults(evms[i], title=modelnames[i])

print('Done')
