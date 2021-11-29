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

teamTricodes = {
    'Hawks': 'ATL', 'Nets': 'BKN', 'Celtics': 'BOS',
    'Hornets': 'CHA', 'Bulls': 'CHI', 'Cavaliers': 'CLE',
    'Mavericks': 'DAL', 'Nuggets': 'DEN', 'Pistons': 'DET',
    'Warriors': 'GSW', 'Rockets': 'HOU', 'Pacers': 'IND',
    'Clippers': 'LAC', 'Lakers': 'LAL', 'Grizzlies': 'MEM',
    'Heat': 'MIA', 'Bucks': 'MIL', 'Timberwolves': 'MIN',
    'Pelicans': 'NOP', 'Knicks': 'NYK', 'Thunder': 'OKC',
    'Magic': 'ORL', '76ers': 'PHI', 'Suns': 'PHX',
    'Blazers': 'POR', 'Kings': 'SAC', 'Spurs': 'SAS',
    'Raptors': 'TOR', 'Jazz': 'UTA', 'Wizards': 'WAS'
}

seasonDates = {
    '13/14': ('2013-10-09', '2014-04-16'),
    '14/15': ('2014-10-28', '2015-04-15'),
    '15/16': ('2015-10-27', '2016-4-13'),
    '16/17': ('2016-10-25', '2017-04-12'),
    '17/18': ('2017-10-17', '2018-04-11'),
    '18/19': ('2018-10-16', '2019-04-10'),
    '19/20': ('2019-10-22', '2020-08-15'),
}

# --------------


def importData(filedir, filename):
    """
    Wrapper function for pd.read_csv().
    """
    print(
        f"-- Imported '{filename}' from the '{path.relpath(filedir)}' directory! --'")
    return pd.read_csv(path.join(filedir, filename))


def exportData(df, filedir, filename):
    """
    Wrapper function for pd.to_csv().
    """
    print(
        f'-- Exported {filename} to the {path.relpath(filedir)} directory! --')
    df.to_csv(path.join(filedir, filename), index=False)


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
            folder
        ),
        f'{folder}{season}.csv'
    )


def insertChar(s, ind=2, sep='/'):
    """
    Inserts 'sep' in 's' at index 'ind'.
    """
    return s[:ind] + sep + s[ind:]


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

        seasondf['Season'] = insertChar(season, ind=2, sep='/')
        frames.append(seasondf)
    return pd.concat(frames)


def computeStatTotals(df, cols):
    """
    Computes total values of stats and
    adds them to dataframe.
    """
    for stat in cols:
        df[f'{stat}-PG'] = df[stat]  # save per-game column
        df[f'{stat}-TOT'] = df[stat].multiply(df['GP'], axis='index')
    df = df.drop(columns=cols)
    return df


def outputFullStats(stats, seasons):
    """
    For each stat, this function outputs
    the data from 2013-2021 to the 'assets/processed'
    directory.
    """
    for stat in stats:
        df = concatSeasons(stat[0], seasons)
        df = df.drop(columns=stat[1])
        # change col names and convert distance values to Km or metres
        if stat[0] == 'speed&distance':
            df = df.rename(
                columns={
                    'Dist. Miles': 'Dist',
                    'Dist. Miles Off': 'Dist. Off',
                    'Dist. Miles Def': 'Dist. Def',
                    'Avg Speed': 'Avg Speed',
                    'Avg Speed Off': 'Avg Speed Off',
                    'Avg Speed Def': 'Avg Speed Def'
                })
            for col in ['Dist', 'Dist. Off', 'Dist. Def', 'Avg Speed', 'Avg Speed Off', 'Avg Speed Def']:
                df[col] = milesToKm(df[col])
        if stat[0] == 'rebounds':
            df = df.rename(
                columns={
                    'DeferredREB Chances': 'DeferredREB Chances',
                    'AVG REBDistance': 'AVG REBDistance',
                }
            )
            df['AVG REBDistance'] = feetToMetres(df['AVG REBDistance'])
        exportData(
            df,
            processedDataDir,
            f'{stat[0]}.csv'
        )
    return 1


def splitDate(df):
    """
    Change Date column format to
    separate columns with yy, mm, dd.
    """
    datesDf = pd.DataFrame(
        df['Date'].str.split('-', 2).to_list(),
        columns=['Year', 'Month', 'Day']
    )
    df = pd.concat([datesDf, df], axis=1)
    return df


def concatStats(dataframes, mergeOn):
    """
    Returns a single DataFrame with all the relevant NBA stats
    scraped from their website.
    """
    return reduce(lambda left, right: pd.merge(
        left, right, on=mergeOn, how='inner'
    ), dataframes)


def getBodyMetrics(filename, dir=rawDataDir):
    df = importData(dir, filename)[[
        'season',
        'player_name',
        'team_abbreviation',
        'age',
        'player_height',
        'player_weight',
    ]].rename(
        columns={
            'season': 'Season',
            'player_name': 'Player',
            'team_abbreviation': 'Team',
            'age': 'Age',
            'player_height': 'Height',
            'player_weight': 'Weight',
        }
    )
    df['Age'] = df['Age'].astype(int)
    # filter out non-relevant seasons
    seasonFilter = np.where(
        (df['Season'].str.split('-').str[0] >= '2013') &
        (df['Season'].str.split('-').str[0] <= '2019')
    )
    df['Season'] = df['Season'].str[2:].replace('-', '/', regex=True)
    return df.iloc[seasonFilter].sort_values(by=['Season', 'Player']).reset_index(drop=True)


def setTeamId(df, teams=teamTricodes):
    """
    Change Team name to its tri-code
    i.e -- Bulls -> CHI
    """
    for name, tricode in teams.items():
        df['Team'] = df['Team'].replace(
            name,
            tricode
        )
    return df


def splitInjuriesIntoSeasons(df, seasonDates):
    # temporarily disable chained assignment warning
    pd.options.mode.chained_assignment = None

    allSeasonsDfs = []
    # create new column with dummy values
    df['Season'] = '-'
    for season, dates in seasonDates.items():
        startDate = dates[0]
        endDate = dates[1]
        seasonInjuriesFilter = np.where(
            (df['Date'] >= startDate) &
            (df['Date'] <= endDate)
        )[0]
        seasonData = df.iloc[seasonInjuriesFilter]
        seasonData['Season'] = season
        allSeasonsDfs += [seasonData]
    # turn warning back on
    pd.options.mode.chained_assignment = 'warn'
    return pd.concat(allSeasonsDfs).drop(columns=['Date']).reset_index(drop=True)


def preprocessInjuries(df, seasonDates=seasonDates):
    """
    Removes players that returned from injury.
    Splits date into yy-mm-dd.
    """
    # Drops the lines where the 'Relinquished'
    # column in the 'injuries' dataset is equal to
    # NaN, i.e player was activated from IL.
    # 'Acquired' column is full of NaN's, so we drop it
    df = df.drop(
        np.where(pd.isnull(df['Relinquished']))[0]
    ).reset_index(drop=True)
    df = df.drop(columns=['Acquired'])

    # for readability and coherence with the other datasets
    df = df.rename(columns={'Relinquished': 'Player'})
    df = splitDate(df)
    df = setTeamId(df)

    # @TODO - Remove data until 2013-10-29 (start of the '13-'14 season)
    # pq só temos info acerca das outras stats
    # a partir do inicio dessa época
    df = splitInjuriesIntoSeasons(df, seasonDates)

    return df
    # pass


def seriesToFrame(series, columns):
    """
    Transforms a Pandas Series object
    into a Dataframe object with the
    given column name(s).
    """
    frame = pd.DataFrame(series).reset_index()
    frame.columns = columns
    return frame


def findInNotes(notes, keyword):
    """
    Finds specific keyword in the 'Notes'
    column of the injuries dataset.
    """
    return notes.str.contains(keyword, regex=True)


def milesToKm(df):
    return df * 1.60934


def feetToMetres(df):
    return df * 0.3048


# Statistical methods
def normalize(df):
    return (df - df.mean()) / df.std()


def mean(df):
    return df.mean()


def median(df):
    return df.median()


def variance(df):
    return df.var()


def skewness(df):
    return df.skew()


def kurtosis(df):
    return df.kurtosis()


def iqr(df):
    return df.quantile(.75) - df.quantile(.25)


def zscore(df, k):
    zs = abs((df - df.mean()) / df.std())
    return df[zs < k], df[zs >= k]


def zcr(df):
    return ((df[:-1] * df[1:]) < 0).sum()


def getStatsAllSeasons(df, seasons):
    cols = list(
        set(df.columns.values.tolist()) -
        set(['Player', 'Team', 'Season', 'Age'])
    )
    seasonStats = {}
    for season in seasons:
        s = insertChar(season, ind=2, sep='/')
        seasonFilter = np.where(df['Season'] == s)[0]
        for i in range(len(cols)):
            seasonStats[f'{s}-{cols[i]}'] = getStatsBySeason(
                df,
                seasonFilter,
                list(df.columns.values).index(cols[i]))
    return pd.DataFrame.from_dict(seasonStats, orient='index')


def getStatsBySeason(df, seasonFilter, colInd):
    seasonDf = normalize(df.iloc[seasonFilter, colInd])
    return {
        'mean': mean(seasonDf),
        'median': median(seasonDf),
        'variance': variance(seasonDf),
        'skewness': skewness(seasonDf),
        'kurtosis': kurtosis(seasonDf),
        'iqr': iqr(seasonDf),
        'zcr': zcr(seasonDf)
    }
