"""
In this file, all the imports that are needed
throughout the program are defined here.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos: (???)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from scipy.stats import linregress
from scipy.stats import spearmanr
from os import path, listdir
from functools import reduce
import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
