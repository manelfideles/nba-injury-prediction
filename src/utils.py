"""
In this file reside all the predictor's
functional components.
Pre-processing, exploratory analysis and implementation
of the predictor are all in this file.

To run any of these functions, please 'cd' into
the 'src' directory.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

import pandas as pd
import matplotlib as plt
import numpy as np
from os import path

# -- globals
# -- these can be accessed from
# -- the main file, so there's no need to
# -- define them there.

rawDataDir = path.realpath('../assets/raw')
processedDataDir = path.realpath('../assets/processed')
# ----------


def importRawData(path):
    """
    Wrapper function for pd.read_csv().
    """
    return pd.read_csv(path)


def exportProcessedData(df, path):
    """
    Wrapper function for pd.to_csv().
    """
    df.to_csv(path, index=False)


def importTrackingData(folder, season):
    """
    Outputs a dataframe with the contents
    of directory with path
    'rawDataDir/folder/player_tracking_folder_season'.
    'season' argument must be in a 'yy''yy' 
    format, i.e drives1314 is the dataset relative to
    drives per game in the '13-'14 season.
    """
    return importRawData(
        path.join(
            rawDataDir,
            folder,
            f'{folder}{season}.csv'
        )
    )


def concatSeasons(stat, seasons):
    frames = []
    for season in seasons:
        seasondf = importTrackingData(stat, season)
        seasondf['Season'] = season
        frames.append(seasondf)
    return pd.concat(frames)


def outputFullStats(stats, seasons):
    """
    For each stat, this function outputs
    the data from 2013-2021 to the 'assets/processed'
    directory.
    """
    for stat in stats:
        df = concatSeasons(stat, seasons)
        exportProcessedData(df, path.join(
            processedDataDir, f'{stat}.csv'
        ))
    return 0
