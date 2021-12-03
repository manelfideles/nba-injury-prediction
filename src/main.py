"""
This is the main file of
the predictor.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

from dependencies import *
from utils import *
from tests import *


# -- globals
seasons = [
    '1314', '1415', '1516', '1617',
    '1718', '1819', '1920'
]

# (stat, [columns_to_exclude])
stats = [
    ('drives', ['W', 'L', 'FGM', 'FGA', 'FG%', 'FTM', 'FTA', 'FT%', 'PTS',
                'PTS%', 'PASS', 'PASS%', 'AST', 'AST%', 'TO', 'TOV%', 'PF', 'PF%']),
    ('fga', ['W', 'L', 'FGM', 'FG%', '3PM', '3P%', 'PTS', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB',
             'AST', 'TOV', 'STL', 'BLK', 'PF', 'FP', 'DD2', 'TD3', '+/-']),
    ('rebounds', ['W', 'L', 'ContestedREB%',
                  'REBChance%', 'AdjustedREB Chance%']),
    ('speed&distance', ['W', 'L', 'Dist. Feet']),
]

debug = False
injuries_eda = False
stats_eda = True
# ----------


# -- Dataset generation and loading
# 'injuries.csv'
if not path.isfile(path.join(processedDataDir, 'injuries.csv')):
    exportData(
        preprocessInjuries(
            importData(rawDataDir, f'og_injuries.csv'),
        ),
        processedDataDir,
        'injuries.csv'
    )


# 'stats.csv'
if len(listdir(processedDataDir)) <= len(stats) + 2:
    # generate player tracking datasets
    outputFullStats(stats, seasons)
    # import somewhat pre-processed player tracking data
    drives = importData(processedDataDir, 'drives.csv')
    fga = importData(processedDataDir, 'fga.csv').drop(columns=['Age'])
    rebounds = importData(processedDataDir, 'rebounds.csv')
    speed_distance = importData(processedDataDir, 'speed&distance.csv')

    # eliminate Tyler Ulis
    drives = drives.drop(np.where(drives['Player'] == 'Tyler Ulis')[0], axis=0)
    fga = fga.drop(np.where(fga['Player'] == 'Tyler Ulis')[0], axis=0)
    rebounds = rebounds.drop(
        np.where(rebounds['Player'] == 'Tyler Ulis')[0], axis=0)
    speed_distance = speed_distance.drop(
        np.where(speed_distance['Player'] == 'Tyler Ulis')[0], axis=0)

    getBodyMetrics('body_metrics.csv', rawDataDir)
    body_metrics = importData(processedDataDir, 'bodymet.csv')
    body_metrics = body_metrics.drop(
        np.where(body_metrics['Player'] == 'Tyler Ulis')[0], axis=0)
    commonData = drives['Player'], drives['Team'], drives['GP'], drives['MIN'], drives['Season']
    commonData = concatStats(commonData)

    # generate final dataset with all relevant tracking data
    statsToCompute = [
        'MIN', 'DRIVES', 'FGA', '3PA',
        'REB', 'ContestedREB', 'REBChances', 'DeferredREB Chances',
        'Dist', 'Dist. Off', 'Dist. Def'
    ]
    cs = concatStats(
        [drives, fga, speed_distance, rebounds],
        ignore=['Player', 'Team', 'GP', 'MIN', 'Season']
    )
    cs2 = concatStats([commonData, cs, body_metrics.drop(
        columns=['Player', 'Team', 'Season'])])
    st = computeStatTotals(cs2, statsToCompute)
    st['BMI'] = calculatePlayerBMI(st['Height'], st['Weight'])
    exportData(st.sort_values(
        by=['Season', 'Player']), processedDataDir, 'stats.csv')

# 'travels.csv'
if not path.isfile(path.join(processedDataDir, 'travels.csv')):
    processTravelData(
        sanitizeTravelMetrics(rawDataDir, 'travel_metrics.csv')
    )
# ---------------------

# Exploratory Data Analysis on the injuries dataset
if injuries_eda:
    injuries = importData(processedDataDir, 'injuries.csv')
    # eliminate Tyler Ulis entries
    tylerulis = np.where(injuries['Player'] == 'Tyler Ulis')[0]
    injuries = injuries.drop(tylerulis, axis=0)
    # 1 -- Teams with the most injuries
    # !! - Limited to the top 'limit' most injured teams
    topInjuriesTeams = seriesToFrame(
        injuries['Team'].value_counts(),
        ['Team', '# of Events']
    ).sort_index()
    plotHistogram(
        topInjuriesTeams,
        topInjuriesTeams.columns,
        limit=10,
        title='Injuries by team'
    )
    # 2 -- Players with the most injuries
    # !! - Limited to the top 'limit' most injured players
    topInjuriesPlayers = seriesToFrame(
        injuries['Player'].value_counts(),
        ['Player', '# of Events']
    )
    plotHistogram(
        topInjuriesPlayers,
        topInjuriesPlayers.columns,
        limit=10,
        title='Injuries by player'
    )

    # 3 -- @TODO Injuries per player

    # 4 -- Injuries per year
    topInjuriesYear = seriesToFrame(
        injuries['Year'].value_counts(),
        ['Year', '# of Events']
    ).sort_values('Year').reset_index(drop=True)
    plotLineGraph(
        topInjuriesYear,
        topInjuriesYear.columns,
        title='Injuries over the years'
    )

    # 5 -- Injuries per month
    topInjuriesPerMonth = seriesToFrame(
        injuries['Month'].value_counts(),
        ['Month', '# of Events']
    ).sort_values('Month').reset_index(drop=True)
    plotLineGraph(
        topInjuriesPerMonth,
        topInjuriesPerMonth.columns,
        title='Injuries per month'
    )

    # Injury types
    # 1 -- Injuries per type of injury
    topInjuriesByType = seriesToFrame(
        injuries['Notes'].value_counts(),
        ['Types', '# of Events']
    )
    plotHistogram(
        topInjuriesByType,
        topInjuriesByType.columns,
        limit=40,
        title='# of Injuries per type'
    )

    # 2 -- # rest entries vs non-rest entries
    restEntries = seriesToFrame(
        findInNotes(injuries['Notes'], 'rest').value_counts(),
        ['Rest?', '# of Events']
    )

    restFilter = np.where((findInNotes(injuries['Notes'], 'rest') == True))[0]
    # 3 -- Resting over the years
    restYears = seriesToFrame(
        injuries.iloc[restFilter, 0].value_counts(),  # column 0 == 'Year'
        ['Year', '# of Events']
    ).sort_values('Year').reset_index(drop=True)
    plotLineGraph(
        restYears,
        restYears.columns,
        title='Resting over the years'
    )

    # 4 -- Resting over months
    restMonths = seriesToFrame(
        injuries.iloc[restFilter, 1].value_counts(),  # column 1 == 'Month'
        ['Month', '# of Events']
    ).sort_values('Month').reset_index(drop=True)
    plotHistogram(
        restMonths,
        restMonths.columns,
        limit=12,
        orientation='vert',
        title='Resting over the months'
    )

    # 5 -- Resting by team
    restTeams = seriesToFrame(
        injuries.iloc[restFilter, 3].value_counts(),  # column 3 == 'Team'
        ['Team', '# of Events']
    ).sort_values('# of Events')
    plotHistogram(
        restTeams,
        restTeams.columns,
        limit=32,
        title='Resting by team'
    )

    # 6 -- Resting vs Injuries
    topInjuriesPlayers = seriesToFrame(
        injuries['Player'].value_counts(),
        ['Player', '# of Events']
    )

    restFilter = np.where((findInNotes(injuries['Notes'], 'rest') == True))[0]
    restPlayers = seriesToFrame(
        injuries.iloc[restFilter, 4].value_counts(),  # column 4 == 'Player
        ['Player', '# of Events']
    ).sort_index()

    restingVsInjuries = restPlayers.merge(
        topInjuriesPlayers,
        how='inner',
        on='Player'
    ).rename(columns={
        '# of Events_x': '# of Rests',
        '# of Events_y': '# of Injuries'
    })

    mostInjuredPlayer = topInjuriesPlayers.iloc[0, :]['Player']
    mostRestedPlayer = restPlayers.iloc[0, :]['Player']

    plotScatterGraph(
        restingVsInjuries,
        ['# of Injuries', '# of Rests'],
        'Relationship between injuries and rest (by player)',
        [topInjuriesPlayers.iloc[0, :]['Player'], restPlayers.iloc[0, :]['Player']]
    )

    # 7 -- DNP, DTD, out indefinitely, out for season, other
    status = ['DNP', 'DTD', 'out indefinitely', 'out for season']
    countSeverities = {
        s: np.where(
            findInNotes(
                injuries['Notes'], s
            ) == True
        )[0].size for s in status
    }

    injurySeverities = pd.DataFrame(countSeverities, index=[0]).T.rename(
        columns={0: '# of Events'}
    ).sort_index()

    plotHistogram(
        injurySeverities,
        [None, '# of Events'],
        title='Severity of injuries and their frequencies',
        dim=1
    )

statsDataset = importData(processedDataDir, 'stats.csv')
print(statsDataset)

travelDataset = importData(processedDataDir, 'travels.csv')

print(travelDataset)
c = 0
for p1, p2 in zip(statsDataset['Player'], travelDataset['Player']):
    if p1 != p2:
        print(c, p1, p2)
        break
    else:
        c += 1


# para cada jogador:
#   - contar o # de injuries que teve naquela epoca
#   - contar o # de rests que teve naquela epoca
# adicionar ao dataset de stats


print('Done')
