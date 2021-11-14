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

from dependencies import *

# -- globals
# -- these can be accessed from
# -- the main file, so there's no need to
# -- define them there.

rawDataDir = path.realpath('../assets/raw')
processedDataDir = path.realpath('../assets/processed')
# ----------


def importData(path):
    """
    Wrapper function for pd.read_csv().
    """
    return pd.read_csv(path)


def exportData(df, path):
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
    return importData(
        path.join(
            rawDataDir,
            folder,
            f'{folder}{season}.csv'
        )
    )


def concatSeasons(stat, seasons):
    """
    Concatenates stat data relative to all
    seasons.
    """
    frames = []
    for season in seasons:
        seasondf = importTrackingData(stat, season)

        # remove a lot of unnecessary empty
        # columns
        if stat == 'fga':
            seasondf = seasondf.loc[:, ~seasondf.columns.str.endswith('RANK')]

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
        exportData(df, path.join(
            processedDataDir, f'{stat}.csv'
        ))
    return 0


def removeAcquired(df):
    """
    Removes players that returned from injury.
    This function drops the lines where the 'Relinquished'
    column in the 'injuries' dataset is equal to
    NaN, i.e a player was activated from IL.
    After this, the 'Acquired' column no longer has meaningful data,
    so we can also drop it.
    """
    df = df.drop(
        np.where(pd.isnull(df['Relinquished']))[0]
    )
    return df.drop(columns=['Acquired'])
