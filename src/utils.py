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
from tests import plotMultiple
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(0)

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


def concatStats(dfs, ignore=None, axis=1):
    """
    Returns a single DataFrame with all the relevant NBA stats
    scraped from their website.
    """
    if ignore:
        dfs = [df.drop(columns=ignore) for df in dfs]
    return pd.concat(dfs, join='outer', axis=axis)


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
        (df['Season'].str.split('-').str[0] <= '2018')
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


def sanitizeTravelMetrics(dir, filename):
    df = importData(dir, filename)[[
        'Season',
        'Player',
        'Team',
        'Date',
        'Distance',
        'Flight Time',
        'Shift (hrs)'
    ]].rename(
        columns={
            'Distance': 'Distance Travelled',
            'Shift (hrs)': 'TZ Shift (hrs)'
        })
    df = splitDate(df).drop(['Date', 'Year', 'Day'], axis=1)
    df['Season'] = df['Season'].str[2:].replace('-', '/', regex=True)
    df['Month'] = df['Month'].astype(int)
    df = setTeamId(df, teams=teamTricodes)
    seasonFilter = np.where(
        (df['Season'] == '18/19') &
        (df['Month'] == 4)
    )[0]
    df = df.iloc[:seasonFilter[-1] + 1]
    return df.sort_values(by=['Season', 'Month', 'Player'])


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


def makeStatsDataset(seasons, monthNumbers, stats):
    allstats = []
    for stat in stats.keys():
        stats_out = []
        for season in seasons:
            for monthname, monthnumber in monthNumbers.items():
                data = importData(
                    path.join(rawDataDir, stat, 'monthly'),
                    f'{stat}_{monthname}_{season}.csv'
                )
                data.columns = data.columns.str.replace(
                    ' ', '')
                data = data.rename(
                    columns={'PLAYER': 'Player', 'TEAM': 'Team', 'AGE': 'Age'})
                data = data[stats[stat]]
                data['Month'] = monthnumber
                data['Season'] = insertChar(season)
                data['Player'] = removeDotsInPlayerName(data['Player'])
                stats_out.append(data)
        stat_out = pd.concat(stats_out) \
            .sort_values(by=['Season', 'Player']) \
            .reset_index(drop=True)
        allstats.append(stat_out)
    exportData(allstats[0],
               processedDataDir, 'monthly_drives.csv')
    exportData(allstats[1],
               processedDataDir, 'monthly_fga.csv')
    exportData(allstats[2],
               processedDataDir, 'monthly_rebs.csv')
    exportData(allstats[3],
               processedDataDir, 'monthly_sd.csv')


def concatInjury(df, injdf, monthNumbersRev):
    df['Injured'] = 0
    for player, team, month, season in zip(df['Player'], df['Team'], df['Month'], df['Season']):
        # print(player, team, month, season)
        injfilter = np.where(
            (injdf['Player'] == player) & (injdf['Team'] == team) &
            (injdf['Month'] == monthNumbersRev[month]) &
            (injdf['Season'] == season)
        )[0]
        if len(injfilter):
            currentPlayer = np.where(
                (df['Player'] == player) & (df['Team'] == team) &
                (df['Month'] == month) & (
                    df['Season'] == season)
            )[0]
            df.loc[currentPlayer, 'Injured'] = 1
    return df


def concatTravels(df, tdf, monthNumbersRev):
    df['Distance Travelled-TOT'] = 0
    df['Distance Travelled-PG'] = 0
    df['TZ Shift-TOT'] = 0
    df['TZ Shift-PG'] = 0
    for player, team, month, season in zip(df['Player'], df['Team'], df['Month'], df['Season']):
        distFilter = np.where(
            (tdf['Player'] == player) & (tdf['Team'] == team) &
            (tdf['Month'] == monthNumbersRev[month]) & (
                tdf['Season'] == season)
        )[0]
        # print(player, team, f'{month}({monthNumbersRev[month]})', season, distFilter)
        if len(distFilter):
            currentPlayer = np.where(
                (df['Player'] == player) & (df['Team'] == team) &
                (df['Month'] == month) & (
                    df['Season'] == season)
            )[0]
            df.loc[currentPlayer, 'Distance Travelled-TOT'] = tdf.iloc[
                distFilter, -3
            ].sum()
            df.loc[currentPlayer, 'Distance Travelled-PG'] = tdf.iloc[
                distFilter, -3
            ].mean()
            df.loc[currentPlayer, 'TZ Shift-TOT'] = tdf.iloc[
                distFilter, -1
            ].abs().sum()
            df.loc[currentPlayer, 'TZ Shift-PG'] = tdf.iloc[
                distFilter, -1
            ].abs().mean()
            # print(df.loc[currentPlayer, 'TZ Shift-PG'])
    return df


# ---------------------------------
# Feature Selection, training and testing


def doPca(dataset, n_components):
    pca = PCA(n_components=n_components)
    pca_dataset = pca.fit_transform(dataset)
    return pca, pca_dataset


def getEvrs(dataset):
    """
    Returns EVRs for a different
    number of PC's
    """
    evratios = []
    for i in range(len(dataset.columns)):
        pca, _ = doPca(dataset, i)
        evratios.append(sum(pca.explained_variance_ratio_) * 100)
    return evratios


def findPCs(evratios, accuracy):
    """
    Finds index of number
    of pc's to include in our pca
    by intersecting the evr %
    and the evr's of pca.
    """
    return np.argwhere(np.diff(np.sign(
        evratios - np.repeat(accuracy, len(evratios))
    ))).flatten()[0]


def plotEvrPc(evratios, pc):
    _, ax = plt.subplots()
    x = np.arange(len(evratios))
    ax.plot(x, evratios)
    ax.plot(x[pc], evratios[pc], 'x', markersize=12,
            label=f'# of PCs: {x[pc]}')
    ax.axhline(evratios[pc], color='r', ls='--',
               label=f'{round(evratios[pc], 2)} % EVR')
    ax.legend(loc='lower right')
    ax.set(
        xlabel='number of PCs',
        ylabel='explained variance ratio (%)',
        title='# of PCs vs EVR'
    )
    ax.grid()
    plt.show()


def savePCAFeatures(model, cols, n_components):
    """
    Stores the 'n_components' features
    with the most EVR.
    """
    most_important_pcs = [
        np.abs(model.components_[i]).argmax() for i in range(n_components)
    ]
    most_important_names = [
        cols[most_important_pcs[i]] for i in range(n_components)
    ]
    return list(set([most_important_names[i] for i in range(n_components)]))


def getStatisticalMetrics(df):
    return df.agg([
        'mean', 'median', 'std',
        'var', 'skew', 'kurtosis',
        lambda x: x.quantile(.75) - x.quantile(.25)
    ]).rename(index={'<lambda>': 'iqr(75/25)'})


# Splitting
def tt(data, target, testsize=0.3):
    """
    Wrapper function for train_test_split.
    Returns 4 values:
    - data_train and data_test; target_train and target_test.
    Follows the 70%-30% rule-of-thumb for TT
    by default.
    """
    return tts(data, target, test_size=testsize, random_state=42)


def getEvaluationMetrics(target, prediction):
    # https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/
    """
    Confusion Matrix:
    Precision (Prec): #Tp / (#Tp + Fp)
    Recall (Rec): #Tp / (#Tp + Fn)
    Accuracy (Acc): Correct Predictions/Total predictions
    Balanced Accuracy (BAcc): Acc / #classes
    F1: 2 * (P*R)/(P+R)
    NOTE: One limitation of these metrics is that they
    assume that the class distribution observed in the
    training dataset will match the distribution in the
    test set and in real data when the model is used
    to make predictions. This is often the case,
    but when it is not the case, the performance can
    be quite misleading.
    """
    precision, recall, _ = precision_recall_curve(target, prediction)
    return confusion_matrix(target, prediction), \
        round(recall_score(target, prediction), 4), \
        round(precision_score(target, prediction), 4), \
        round(accuracy_score(target, prediction), 4), \
        round(accuracy_score(target, prediction), 4) / 2, \
        round(f1_score(target, prediction), 4), \
        round(roc_auc_score(target, prediction, average='weighted'), 4), \
        auc(recall, precision)


def reportEvaluationMetrics(args):
    cm, r, p, a, f1 = args
    return """Confusion Matrix: {}
    Recall Score: {}
    Precision Score: {}
    Accuracy Score: {}
    F1 Score: {}
    """.format(cm, r, p, a, f1)


def ttSplit(data, target, testsize, balanced=False, playercol=None, seasoncol=None):
    """
    Splits dataset into training and
    If 'balance' is True, then data from each player is split
    and distributed among the train and test sets.
    If 'balance' is False, then the testing set will
    contain unseen data, which can hamper the model's performance.
    This argument is good for testing the generalizability of the model.
    Returns data_train, data_test, target_train, target_test
    """
    if balanced:
        data['Player'] = playercol
        data['Season'] = seasoncol
        data['Target'] = target
        train, test = [], []
        for _, obs in data.groupby(['Player', 'Season']):
            obs = obs.reset_index(drop=True)
            if len(obs) > 1:
                test_filter = obs.sample(
                    math.ceil(len(obs) * testsize),
                    replace=True).index.values
                train_filter = np.delete(
                    obs.index.values,
                    np.where(obs.index.values == test_filter)
                )
                obs = obs.drop(['Player', 'Season'], axis=1)
                train += [obs.iloc[train_filter]]
                test += [obs.iloc[test_filter]]

        return pd.concat(train).reset_index(drop=True).iloc[:, :-1], \
            pd.concat(test).reset_index(drop=True).iloc[:, :-1], \
            pd.concat(train).reset_index(drop=True).iloc[:, -1], \
            pd.concat(test).reset_index(drop=True).iloc[:, -1]
    else:
        return tt(data, target, testsize=testsize)


def flattenVec(observations, winsize, increment):
    """
    Flattens player observations into
    ['winsize' * len(features)]-sized arrays.
    Returns NaN-padded array if the number of
    player observations is smaller than the window size.
    """
    # remove player, team, season
    observations = [obs[3:] for obs in observations]
    fv = None
    if len(observations) < winsize:
        fv = np.asarray(observations).reshape(-1)
        return np.pad(
            fv, (0, winsize * len(observations[0]) - len(fv)),
            mode='constant', constant_values=np.nan
        )
    else:
        fv = np.hstack((observations[:winsize]))
        for i in range(1, (len(observations) - winsize) + 1, increment):
            fv = np.vstack((fv, np.hstack((observations[i:i+winsize]))))
    return fv


def renameColsWithDupes(df):
    """
    Adds sufix (col name count) to duplicated col names,
    i.e: Injured1, Injured2, Injured3 and reorders cols to
    the format [Player, Season, ...]
    """
    identifier = df.columns.to_series().groupby(level=0).cumcount() + 1
    df.columns = df.columns.astype(str) + '_' + identifier.astype(str)
    return df[['Player_1', 'Season_1'] + df.columns.tolist()[2:-2]].rename(
        columns={'Player_1': 'Player', 'Season_1': 'Season'}
    )


def makeFeatureVectors(df, winsize, increment):
    """
    Returns feature vectors for a player,
    given 'winsize' previous entries in the dataset.
    'increment' controls the overlap of the sliding window.
    """
    windows = []
    for key, item in df.groupby(['Season', 'Player']):
        fv = flattenVec(item.to_numpy(), winsize, increment)
        print(key, fv.shape)
        # player has more than one observation
        if fv.shape != (len(df.columns.tolist()[3:] * winsize),):
            out = pd.DataFrame(fv, columns=df.columns.tolist()[3:] * winsize)
        else:
            out = pd.DataFrame(
                [fv.tolist()],
                columns=df.columns.tolist()[3:] * winsize
            )
        out['Player'], out['Season'] = key[0], key[1]
        windows += [renameColsWithDupes(out)]
    return pd.concat(windows, axis=0)


def selectFeatures(data, target, n_feats, type='classif'):
    """
    Feature Selection for
    numerical input and numerical output.
    https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/.
    Criteria:
    - Regression: Pearson's corrcoef (Univariate feature selection)
    - Classification: ANOVA F-Value (Univariate feature selection)
    'k' is the # of features to select.
    """
    if type == 'regress':
        fs = SelectKBest(score_func=f_regression, k=n_feats)
    elif type == 'classif':
        fs = SelectKBest(score_func=f_classif, k=n_feats)
    return pd.DataFrame(fs.fit_transform(data, target), columns=data.columns[fs.get_support(indices=True)])


def getModelRegressor(model):
    if model == 'linreg':
        return LinearRegression()
    elif model == 'tree':
        return DecisionTreeRegressor(random_state=42)
    elif model == 'forest':
        return RandomForestRegressor(random_state=42)
    elif model == 'lasso':
        return Lasso(alpha=2.5, selection='random', tol=1e-3, random_state=42)
    elif model == 'kn':
        return KNeighborsRegressor(n_neighbors=10)
    elif model == 'dummy':
        return DummyRegressor(strategy='mean')


def varyFeatureNumberReg(data, target, modelname, tsize):
    test = {}
    for i in range(len(data.columns), 1, -1):
        # feature selection
        selected = selectFeatures(data, target, type='regress', n_feats=i)

        # data splitting
        data_train, data_test, target_train, target_test = ttSplit(
            selected, target, testsize=tsize
        )

        # training
        model = getModelRegressor(modelname).fit(data_train, target_train)
        pred_target_test = model.predict(data_test)

        # visualize observed vs predicted
        if debug:
            predtargtest = pd.DataFrame(pred_target_test)
            target_test = target_test.reset_index(drop=True).T
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


def getModelClassif(model):
    if model == 'dummy':
        return DummyClassifier()
    elif model == 'no-injury':
        return DummyClassifier(strategy='most_frequent')
    elif model == 'tree':
        return DecisionTreeClassifier(criterion='entropy')
    elif model == 'forest':
        return RandomForestClassifier(n_estimators=150)
    elif model == 'kn':
        return KNeighborsClassifier(n_neighbors=10)
    elif model == 'svm':
        return SVC(class_weight='balanced')
    elif model == 'nb':
        return GaussianNB()
    elif model == 'mlp':
        return MLPClassifier()
    elif model == 'ridge':
        return RidgeClassifierCV(class_weight='balanced')


def varyFeatureNumberClassif(data, target, modelname, tsize, balanced=False, playercol=None, seasoncol=None):
    print('Model: ', modelname)
    test = {}
    for i in range(1, len(data.columns)+1):
        print(f'# of Features: {i}')
        # feature selection
        selected = selectFeatures(data, target, i, type='classif')

        # data splitting
        if balanced:
            data_train, data_test, target_train, target_test = ttSplit(
                selected, target, tsize,
                balanced=True,
                playercol=playercol,
                seasoncol=seasoncol
            )
        else:
            data_train, data_test, target_train, target_test = ttSplit(
                selected, target, testsize=tsize
            )

        # training
        clf = getModelClassif(modelname)
        pred_target_test = clf.fit(
            data_train, target_train).predict(data_test)

        # visualize observed vs predicted
        if debug:
            predtargtest = pd.DataFrame(pred_target_test)
            target_test = target_test.reset_index(drop=True).T
            plotMultiple(
                pd.concat([predtargtest, target_test], axis=1).T.rename(
                    index={0: 'predicted value', 1: 'real value'}
                ),
                'scatter',
                title=f'[{modelname.capitalize()}] Predicted vs Real - Iter #{i}'
            )

        # storing of evaluation metrics
        test[i] = getEvaluationMetrics(target_test, pred_target_test)

    return pd.DataFrame.from_dict(test) \
        .rename(index={
            0: 'confMat', 1: 'Rec', 2: 'Prec',
            3: 'Acc', 4: 'BAcc', 5: 'F1',
            6: 'ROC AUC', 7: 'PR AUC'
        })
