# Example Python program to draw a scatter plot

# for two columns of a multi-column DataFrame

import pandas as pd

import numpy as np

import matplotlib.pyplot as plot

 

# Create an ndarray with three columns and 20 rows

data = np.random.randn(20, 4);

 

# Load data into pandas DataFrame       

dataFrame = pd.DataFrame(data=data, columns=['A', 'B', 'C', 'D']);

 

# Draw a scatter plot

dataFrame.plot.scatter(x='C', y='D', title= "Scatter plot between two columns of a multi-column DataFrame");

plot.show(block=True);