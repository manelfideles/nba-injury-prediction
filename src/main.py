"""
This is the main file of 
the predictor.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

from matplotlib.pyplot import title
from pandas.core import series
from dependencies import *
from utils import *
from tests import *


# -- globals
seasons = [
    '1314', '1415', '1516', '1617',
    '1718', '1819', '1920', '2021'
]

# (stat, [columns_to_exclude])
stats = [
    ('drives', ['W', 'L', 'FGM', 'FGA', 'FG%', 'FTM', 'FTA', 'FT%', 'PTS',
                'PTS%', 'PASS', 'PASS%', 'AST', 'AST%', 'TO', 'TOV%', 'PF', 'PF%']),
    ('fga', ['W', 'L', 'PTS', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB',
             'AST', 'TOV', 'STL', 'BLK', 'PF', 'FP', 'DD2', 'TD3', '+/-']),
    ('rebounds', ['W', 'L']),
    ('speed&distance', ['W', 'L']),
    'og_injuries'
]

months = [
    'Jan', 'Feb', 'Mar',
    'Abr', 'May', 'Jun',
    'Jul', 'Ago', 'Set',
    'Oct', 'Nov', 'Dez'
]

debug = False
# ----------

# generate and save required datasets:
# tracking and injuries
if len(listdir(processedDataDir)) != len(stats):
    if outputFullStats(stats, seasons):
        print('-- Generated datasets! --')

# import pre-pre-processed player tracking data
drives = importData(path.join(processedDataDir, 'drives.csv'))
fga = importData(path.join(processedDataDir, 'fga.csv'))
rebounds = importData(path.join(processedDataDir, 'rebounds.csv'))
speed_distance = importData(path.join(processedDataDir, 'speed&distance.csv'))
injuries = importData(path.join(processedDataDir, 'injuries.csv'))
print('-- Imported datasets! --')

if debug:
    # Exploratory Data Analysis on the injuries dataset
    # 1 -- Teams with the most injuries
    # !! - Limited to the top 'limit' most injured teams
    topInjuriesTeams = seriesToFrame(
        injuries['Team'].value_counts(),
        ['Team', '# of Events']
    ).sort_index(ascending=False)
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

# Statistical analysis

print('Done')
