"""
In this file reside all methods
designed to test the operations
executed on the data.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

from matplotlib.pyplot import legend
from dependencies import *


def plotHistogram(data, labels, limit=10):
    _, ax = plt.subplots()
    ax.barh(data.iloc[:limit+1, 0], data.iloc[:limit+1, 1])
    ax.set_ylabel(labels[0])
    ax.set_xlabel(labels[1])
    plt.show()
