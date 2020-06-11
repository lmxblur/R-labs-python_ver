import numpy as np
import pandas as pd
lottery = pd.read_excel('lottery.xls')
X = lottery['Day_of_year']
Y = lottery['Draft_No']