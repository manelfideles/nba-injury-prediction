"""
In this file reside all methods
designed to test the operations
executed on the data.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

# from matplotlib.pyplot import legend
from matplotlib.pyplot import legend
from dependencies import *
plt.rcParams['font.size'] = '7'


def plotHistogram(data, axlabels, title=None, limit=10, orientation='horz', dim=2):
    _, ax = plt.subplots()
    if dim == 2:
        if orientation == 'vert':
            ax.bar(data.iloc[0:limit, 0], data.iloc[0:limit, 1])
            ax.set_xlabel(axlabels[0])
            ax.set_ylabel(axlabels[1])
        else:
            ax.barh(data.iloc[:limit+1, 0], data.iloc[:limit+1, 1])
            ax.set_xlabel(axlabels[1])
            ax.set_ylabel(axlabels[0])
    if dim == 1:
        ax2 = sns.barplot(x=data.index, y=axlabels[1], data=data)
        # annotation code from https://www.dataquest.io/blog/how-to-plot-a-bar-graph-matplotlib/
        for p in ax2.patches:
            ax2.annotate(int(p.get_height()),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 7), textcoords='offset points')
    if title:
        plt.title(title)
    plt.show()


def plotLineGraph(data, axlabels, title=None):
    _, ax = plt.subplots()
    ax.plot(data.iloc[:, 0], data.iloc[:, 1], 'o-')

    for i, txt in enumerate(data.iloc[:, 1]):
        ax.annotate(txt, (data.iloc[i, 0], data.iloc[i, 1]))

    if axlabels:
        ax.set_xlabel(axlabels[0])
        ax.set_ylabel(axlabels[1])
    if title:
        plt.title(title)
    plt.show()


def plotMultiple(df, graphtype='line', title=None):
    _, ax = plt.subplots()
    for i in range(len(df)):
        if graphtype == 'line':
            line, = ax.plot(df.columns, df.iloc[i, :], '-o')
        elif graphtype == 'scatter':
            line, = ax.plot(df.columns, df.iloc[i, :], 'o')
        line.set_label(df.index[i])
    ax.legend()
    if title:
        plt.title(title)
    plt.show()


# ainda nao faz nada mas há-de ser util
def annotations(ax, data, ann=[]):
    """
    Iterates over 'data' to annotate
    the contents of 'ann' on the set
    of axes 'ax'.
    """
    for i in range(len(ann)):
        for j, txt in enumerate(data):
            ax.annotate(txt, j)
    pass


def plotScatterGraph(data, axlabels, title, annotations=[]):
    _, ax = plt.subplots()
    ax.scatter(data[axlabels[0]], data[axlabels[1]])

    # best fit line
    sns.regplot(ax=ax, x=axlabels[0], y=axlabels[1], data=data)

    # annotates specific data points defined in
    # the 'annotations' argument
    if len(annotations):
        for i in range(len(annotations)):
            for j, txt in enumerate(data.iloc[:, 0]):
                if(data.iloc[j, 0] == annotations[i]):
                    ax.annotate(txt, (data.iloc[j, 2], data.iloc[j, 1]))

    ax.set_xlabel(axlabels[0])
    ax.set_ylabel(axlabels[1])
    plt.xlim([0, max(data[axlabels[0]])+1])
    plt.ylim([0, max(data[axlabels[1]])+1])
    plt.title(title)
    plt.show()


def plotDistribution(row, stats):
    print(stats)
    sns.displot(row, color='skyblue', kind='kde')
    plt.show()


def plotHeatmap(data, method='pearson'):
    corr = data.corr(method=method)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(
            corr, mask=mask, square=True,
            vmin=-1, vmax=1, xticklabels=1,
            yticklabels=1, center=0
        )
    plt.show()


def plotResults(evm, title=None):
    _, ax = plt.subplots()
    for stat in ['Rec', 'Prec', 'Acc', 'BAcc', 'F1', 'ROC AUC', 'PR AUC']:
        line, = ax.plot(
            evm.columns.values, evm.loc[stat])
        line.set_label(stat)
    ax.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
