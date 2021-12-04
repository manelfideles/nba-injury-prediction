"""
This is the main file of
the predictor. All the training
and results are here.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

from sklearn.utils.extmath import fast_logdet
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

# Feature Selection using Pearson's corrcoeff
print(f'Dataset shape before feature selection: {mainDataset.shape}')
print(f'Dataset columns before feature selection: {mainDataset.columns}')

# @TODO - experimentar diferentes valores e ver como a previsao muda
data_selected = selectFeatures(data, target, n_feats=24)
print(f'Dataset shape *after* feature selection: {data_selected.shape}')
print(f'Dataset columns before feature selection: {data_selected.columns}')


# split

# train - (Baseline) dummy Classi

#

print('Done')
