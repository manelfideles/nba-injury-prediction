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

from pandas.core.frame import DataFrame
from scipy.optimize.zeros import results_c
from seaborn.regression import residplot
from sklearn.utils import shuffle
from dependencies import *
from tests import plotMultiple

# -- globals
# -- these can be accessed from
# -- the main file, so there's no need to
# -- define them there.

rawDataDir = path.realpath('./assets/raw')
processedDataDir = path.realpath('./assets/processed')
debug = False

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

# Dataset-generation util functions


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


def insertChar(s, ind=2, sep='/', replace=False):
    """
    Inserts 'sep' in 's' at index 'ind'
    if there's no separator in 's'.
    Otherwise, replaces sep and returns
    """
    if replace:
        return s.replace(s[ind], sep)
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
            df.reset_index(drop=True).sort_values(by=['Season', 'Player']),
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


def concatStats(dfs, ignore=None):
    """
    Returns a single DataFrame with all the relevant NBA stats
    scraped from their website.
    """
    if ignore:
        dfs = [df.drop(columns=ignore) for df in dfs]
    return pd.concat(dfs, join='outer', axis=1)


def getBodyMetrics(filename, dir=rawDataDir):
    df = importData(dir, filename)[[
        'player_name',
        'team_abbreviation',
        'age',
        'player_height',
        'player_weight',
        'season',
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
    df = df.iloc[seasonFilter].reset_index(
        drop=True).sort_values(by=['Season', 'Player'])
    exportData(df, processedDataDir, 'bodymet.csv')


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


def removeDotsInPlayerName(df):
    return df.str.replace('.', '')


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


def calculatePlayerBMI(height, weight):
    return (weight / (height/100) ** 2)


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


def getStatsAllSeasons(df):
    sf = {}
    for col in df.columns:
        if col not in ['Player', 'Team', 'Season']:
            sf[col] = getStatisticalFeatures(df[col])
    return pd.DataFrame.from_dict(sf)


def getStatisticalFeatures(df):
    return {
        'mean': mean(df),
        'median': median(df),
        'variance': variance(df),
        'skewness': skewness(df),
        'kurtosis': kurtosis(df),
        'iqr': iqr(df)
    }


def sanitizeTravelMetrics(dir, filename):
    return importData(dir, filename)[[
        'Season',
        'Player',
        'Team',
        'Distance',
        'Flight Time',
        'Shift (hrs)'
    ]].rename(
        columns={
            'Distance': 'Distance Travelled',
            'Shift (hrs)': 'TZ Shift (hrs)'
        }
    )


def calcDistanceMetrics(df):
    return milesToKm(df['Distance Travelled']) * df['Count']


def calcTzMetrics(df):
    return df['TZ Shift (hrs)'].abs() * df['Count']


def extractTravelMetrics(df, met, cols):
    # group by desired metric (and season and player)
    cnt = pd.DataFrame(
        {'Count': df.groupby(
            ['Player', 'Season', cols[0]]).size()}
    ).sort_values(by=['Season', 'Player']).reset_index()

    # perform necessary changes to dataset
    if met == 'distance':
        cnt[cols[1]] = calcDistanceMetrics(cnt)
    elif met == 'tz':
        cnt[cols[1]] = calcTzMetrics(cnt)
    elif met == 'inj':
        pass
    else:
        pass
    cnt['Season'] = cnt['Season'].str[2:].replace(
        '-', '/', regex=True)

    # drop unnecessary cols and sum values
    cnt = cnt.drop(columns=[cols[0], 'Count'])
    cnt = cnt \
        .groupby(['Player', 'Season'])[cols[1]] \
        .sum() \
        .reset_index()
    cnt = cnt \
        .sort_values(by=['Season', 'Player']) \
        .reset_index(drop=True)
    return cnt


def processTravelData(df):
    # 1 - contar o # de km viajados por
    # cada jogador em cada época
    distanceCnt = extractTravelMetrics(
        df, 'distance', cols=['Distance Travelled', 'Total Distance Travelled (km)'])

    # 2 - contar o # de horas
    # de mudanças de fuso-horário
    tzCnt = extractTravelMetrics(
        df, 'tz', cols=['TZ Shift (hrs)', 'Total TZ Shifts (hrs)'])

    exportData(
        pd.merge(distanceCnt, tzCnt, how='outer', on=['Player', 'Season']),
        processedDataDir,
        'travels.csv'
    )


def getInjuriesPerYear(injuriesDataset, mainDataset, restFilter):
    df = pd.DataFrame(
        {'# of Injuries (Season)': injuriesDataset.groupby(
            ['Player', 'Season']).size()}
    ).sort_values(by=['Season', 'Player']).reset_index()

    mainDataset = getRestsPerYear(injuriesDataset, restFilter, mainDataset)
    mainDataset['# of Injuries (Season)'] = 0
    for player, season, injcnt in zip(df.Player, df.Season, df['# of Injuries (Season)']):
        mainDataset.loc[
            (mainDataset.Player == player.strip()) &
            (mainDataset.Season == season),
            '# of Injuries (Season)'] = injcnt
    mainDataset['# of Injuries (Season)'] -= mainDataset['# of Rests (Season)']
    return mainDataset


def getRestsPerYear(injuriesDataset, restFilter, mainDataset):
    """
    Rests per year per player
    """
    restPlayers = injuriesDataset.iloc[restFilter]
    df = pd.DataFrame(
        {'# of Rests (Season)': restPlayers.groupby(
            ['Player', 'Season']).size()}
    ).sort_values(by=['Season', 'Player']).reset_index()

    mainDataset['# of Rests (Season)'] = 0
    for player, season, rcnt in zip(df.Player, df.Season, df['# of Rests (Season)']):
        mainDataset.loc[
            (mainDataset.Player == player.strip()) &
            (mainDataset.Season == season),
            '# of Rests (Season)'] = rcnt
    return mainDataset


# ---------------------------------
# Feature Selection, training and testing

def selectFeatures(data, target, n_feats=20):
    """
    Feature Selection for
    numerical input and numerical output.
    https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/.
    This method uses Pearson's correlation coefficient method
    to select the most relevant features. 'k' is the # of features to select.
    """
    fs = SelectKBest(score_func=f_regression, k=n_feats)
    return pd.DataFrame(fs.fit_transform(data, target), columns=data.columns[fs.get_support(indices=True)])


def tt(data, target, testsize=0.3):
    """
    Wrapper function for train_test_split.
    Returns 4 values:
    - data_train and data_test; target_train and target_test.
    Follows the 70%-30% rule-of-thumb for TT
    by default.
    """
    return tts(data, target, test_size=testsize)


def getModel(model, n_features):
    if model == 'linreg':
        return LinearRegression()
    elif model == 'tree':
        return DecisionTreeRegressor(criterion='friedman_mse', max_features=n_features)
    elif model == 'forest':
        return RandomForestRegressor(n_estimators=20, criterion='mae', max_features=n_features)
    elif model == 'lasso':
        return Lasso()
    elif model == 'mlp':
        return MLPClassifier()
    elif model == 'kn':
        return KNeighborsRegressor()
    elif model == 'dummy':
        return DummyRegressor()


def varyFeatureNumber(data, target, modelname, tsize):
    test = {}
    for i in range(len(data.columns), 1, -1):
        # feature selection
        selected = selectFeatures(data, target, n_feats=i)

        # data splitting
        data_train, data_test, target_train, target_test = tt(
            selected, target, testsize=tsize
        )

        # training
        model = getModel(modelname, i).fit(data_train, target_train)
        pred_target_test = model.predict(data_test)

        # visualize observed vs predicted
        predtargtest = pd.DataFrame(pred_target_test)
        target_test = target_test.reset_index(drop=True).T
        if debug:
            plotMultiple(
                pd.concat([predtargtest, target_test], axis=1).T.rename(
                    index={0: 'predicted value',
                           '# of Injuries (Season)': 'real value'}
                ),
                'scatter',
                title=f'[{modelname.capitalize()}] Predicted vs Real - Iter #{i}'
            )

        # storing of evaluation metrics
        mse = mean_squared_error(target_test, pred_target_test)
        test[i] = (
            mse,  # mean squared error
            mean_absolute_error(target_test, pred_target_test),
            math.sqrt(mse),  # rmse
            r2_score(target_test, pred_target_test),  # r-squared
            (target_test - pred_target_test).mean(),  # mean residuals
        )

    # https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/
    return pd.DataFrame.from_dict(test) \
        .rename(index={
            0: 'mse', 1: 'mae', 2: 'rmse',
            3: 'r-squared', 4: 'avg residual'
        })
