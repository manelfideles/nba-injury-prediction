"""
In this file reside all methods
designed to test the operations
executed on the data.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

from matplotlib.pyplot import legend
from dependencies import *


def plotHistogram(data):
    _, ax = plt.subplots()
    ax.bar(data.iloc[:, 0], data.iloc[:, 1])
    plt.show()
