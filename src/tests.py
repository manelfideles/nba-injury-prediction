"""
In this file reside all methods
designed to test the operations
executed on the data.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

# from matplotlib.pyplot import legend
from dependencies import *
plt.rcParams['font.size'] = '7'


def plotHistogram(data, axlabels, limit=10, orientation='horz'):
    _, ax = plt.subplots()
    if orientation == 'vert':
        ax.bar(data.iloc[:limit+1, 0], data.iloc[:limit+1, 1])
        ax.set_ylabel(axlabels[1])
        ax.set_xlabel(axlabels[0])
    else:
        ax.barh(data.iloc[:limit+1, 0], data.iloc[:limit+1, 1])
        ax.set_ylabel(axlabels[0])
        ax.set_xlabel(axlabels[1])
    plt.show()


def plotLineGraph(data, axlabels):
    _, ax = plt.subplots()
    ax.plot(data.iloc[:, 0], data.iloc[:, 1], 'o-')
    for i, txt in enumerate(data.iloc[:, 1]):
        ax.annotate(txt, (data.iloc[i, 0], data.iloc[i, 1]))
    ax.set_xlabel(axlabels[0])
    ax.set_ylabel(axlabels[1])
    plt.show()
