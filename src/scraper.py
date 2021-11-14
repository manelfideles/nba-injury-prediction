"""
Module for scraping tracking stats from
the NBA website, using Selenium.

To use this module you have to install 
the "Download table as CSV" extension
from the Chrome Store.

@ Manuel Fideles 2018282990
@ Alexandre Santos (???)

"""

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pyautogui as pyag
from time import sleep

driver = webdriver.Chrome(ChromeDriverManager().install())

menuXPath = '/html/body/main/div/div/div[2]/div/div/nba-stat-table/div[1]/div/div/select'

stats = [
    'drives', 'traditional',  # aka 'fg'
    'rebounding', 'speed-distance'
]

seasons = [
    '2013-14',
    '2014-15',
    '2015-16',
    '2016-17',
    '2017-18',
    '2018-19',
    '2019-20',
    '2020-21',
]


def makeUrl(stat, season):
    return f'https://www.nba.com/stats/players/{stat}/?Season={season}&SeasonType=Regular%20Season&sort=PLAYER_NAME&dir=-1'


def openWebpage(url):
    driver.get(url)
    sleep(5)


def fetchPlayersTable(menuXPath):
    # select 'All' players option
    driver.find_element_by_xpath(menuXPath).click()
    sleep(15)
    pyag.typewrite('a')
    sleep(2)
    pyag.typewrite('enter')
    sleep(10)

    # scroll to table
    pyag.scroll(-20000)


def nameCsv(stat, season):
    pyag.typewrite(f'{stat}_{season}')


def downloadCsv(stat, season):
    # right-click -> download as csv
    pyag.rightClick()
    pyag.typewrite('d')

    # name csv
    nameCsv(stat, season)
    pyag.typewrite('enter')


# -- main --
flag = False
for stat in stats:
    for season in seasons:
        openWebpage(makeUrl(stat, season))
        if not flag:
            driver.find_element_by_id('onetrust-accept-btn-handler').click()
            sleep(1)
        print('done')
        fetchPlayersTable(menuXPath)
        print('done')
        downloadCsv(stat, season)
        print('done')
        flag = True
