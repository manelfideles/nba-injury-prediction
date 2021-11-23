"""
This is the main file of 
the predictor.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

from pandas.core import series
from dependencies import *
from utils import *
from tests import *


# -- globals
seasons = [
    '1314', '1415', '1516', '1617',
    '1718', '1819', '1920', '2021'
]

stats = [
    'drives', 'rebounds', 'speed&distance', 'fga',
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

# Exploratory Data Analysis
if debug:
    # 1 -- Teams with the most injuries
    # !! - Limited to the top 'limit' most injured teams
    topInjuriesTeams = seriesToFrame(
        injuries['Team'].value_counts(),
        ['Team', '# of Injuries']
    ).sort_values('Team').reset_index(drop=True)
    plotHistogram(
        topInjuriesTeams,
        topInjuriesTeams.columns,
        limit=10
    )

    # 2 -- Players with the most injuries
    # !! - Limited to the top 'limit' most injured players
    topInjuriesPlayers = seriesToFrame(
        injuries['Player'].value_counts(),
        ['Player', '# of Injuries']
    )
    plotHistogram(
        topInjuriesPlayers,
        topInjuriesPlayers.columns,
        limit=10
    )

    # 3 -- @TODO Injuries per player

    # 4 -- Injuries per year
    topInjuriesYear = seriesToFrame(
        injuries['Year'].value_counts(),
        ['Year', '# of Injuries']
    ).sort_values('Year').reset_index(drop=True)
    plotLineGraph(
        topInjuriesYear,
        topInjuriesYear.columns,
    )

    # 5 -- Injuries per month
    topInjuriesPerMonth = seriesToFrame(
        injuries['Month'].value_counts(),
        ['Month', '# of Injuries']
    ).sort_values('Month').reset_index(drop=True)
    plotLineGraph(
        topInjuriesPerMonth,
        topInjuriesPerMonth.columns
    )

# Injury types
if debug:
    # 1 -- Injuries per type of injury
    topInjuriesByType = seriesToFrame(
        injuries['Notes'].value_counts(),
        ['Types', '# of Injuries']
    )
    plotHistogram(
        topInjuriesByType,
        topInjuriesByType.columns,
        limit=40
    )

    # 2 -- # rest entries vs non-rest entries
    restEntries = seriesToFrame(
        findInNotes(injuries['Notes'], 'rest').value_counts(),
        ['Rest?', '# of events']
    )

    restFilter = np.where((findInNotes(injuries['Notes'], 'rest') == True))[0]
    # 3 -- Resting over the years
    restYears = seriesToFrame(
        injuries.iloc[restFilter, 0].value_counts(),  # column 0 == 'Year'
        ['Year', '# of Events']
    ).sort_values('Year').reset_index(drop=True)
    # plotLineGraph(restYears, restYears.columns)

    # 4 -- Resting over months
    restMonths = seriesToFrame(
        injuries.iloc[restFilter, 1].value_counts(),  # column 1 == 'Month'
        ['Month', '# of Events']
    ).sort_values('Month').reset_index(drop=True)
    plotHistogram(restMonths, restMonths.columns, limit=12, orientation='vert')

    # 5 -- Resting by team
    restTeams = seriesToFrame(
        injuries.iloc[restFilter, 3].value_counts(),  # column 3 == 'Team'
        ['Team', '# of Events']
    ).sort_values('# of Events')
    plotHistogram(restTeams, restTeams.columns, limit=32)

print('Done')
