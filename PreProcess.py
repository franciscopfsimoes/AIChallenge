import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def Visual(df, tgt):

	print(df.columns)

	print(df[tgt].describe())

	#hist = df[tgt].hist(bins=20)

	sns.distplot(df[tgt], bins = 50)
	'''
	fig, ax = plt.subplots(figsize=(12, 12))

	sns.heatmap(df.corr()[[tgt]].sort_values(tgt), vmax = 1, vmin = -1, cmap ='YlGnBu', annot = True, ax = ax)

	ax.invert_yaxis()

	fig, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(18, 11), sharey=True)
	fig.subplots_adjust(wspace=.2, hspace=.35)
	sns.set(font_scale=2)

	sns.scatterplot(x=df['lstat'], y=df[tgt], ax=ax1[0], color='blue', alpha = 0.5)
	sns.scatterplot(x=df['rm'], y=df[tgt], ax=ax1[1], color='blue', alpha = 0.5)
	
	'''





def Data(tgt):
	df = pd.read_csv('Dataset.csv')

	df.drop('ID', inplace = True, axis=1)



	#Visual(df, tgt)

	plt.show()

	return df
