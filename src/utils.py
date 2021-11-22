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

rawDataDir = path.realpath('./assets/raw')
processedDataDir = path.realpath('./assets/processed')

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
        if stat != 'og_injuries':
            df = concatSeasons(stat, seasons)
            exportData(df, path.join(
                processedDataDir, f'{stat}.csv'
            ))
        # -- export injuries dataset
        else:
            exportData(
                preprocessInjuries(
                    importData(path.join(rawDataDir, f'{stat}.csv'))
                ),
                path.join(processedDataDir, 'injuries.csv')
            )
    return 1


def splitDate(df):
    """
    change Date column to yy-mm-dd
    """
    df['Date'] = df['Date'].str.split('-')
    return df


def preprocessInjuries(df):
    """
    Removes players that returned from injury.
    """
    # Drops the lines where the 'Relinquished'
    # column in the 'injuries' dataset is equal to
    # NaN, i.e a player was activated from IL.
    df = df.drop(
        np.where(pd.isnull(df['Relinquished']))[0]
    )

    # 'Acquired' column is full of NaN's, so we drop it
    df = df.drop(columns=['Acquired'])

    # change 'Relinquished' column to 'Player' for readability
    df = df.rename(columns={'Relinquished': 'Player'})

    # change Date column to yy-mm-dd
    df = splitDate(df)
    return df

    # todo - Remove data until 2013-04-17 (end of '12-'13 season)
    # pq só temos info acerca das outras stats
    # a partir do fim dessa época
    pass


def seriesToFrame(series, columns):
    """
    Transforms a Pandas Series object
    into a Dataframe object with the
    given column name(s).
    """
    frame = pd.DataFrame(series).reset_index()
    frame.columns = columns
    return frame
