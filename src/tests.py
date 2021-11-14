"""
In this file reside all methods
designed to test the operations
executed on the data.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

from dependencies import *


def plotHistogram(df):
    teams = df['Team'].nunique()
    n_injuries = df['Team'].count()
    print(teams, n_injuries)
