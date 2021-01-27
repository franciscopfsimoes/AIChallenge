import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import PreProcess as prepro
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as pca
from sklearn import preprocessing

def analysis(df):

	scaled_df = preprocessing.scale(df)

	pc = pca()

	try:

		pc.fit(scaled_df)

	except:

		pc.fit(scaled_df)

	pca_data = pc.transform(scaled_df)

	per_var = np.round(pc.explained_variance_ratio_ * 100, decimals= 1)

	labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

	plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
	plt.ylabel('Percentage of Explained Variance')
	plt.xlabel('Principal Component')
	plt.title('Scree Plot')
	plt.show()

	pca_df = pd.DataFrame(pca_data, columns=labels)

	plt.scatter(pca_df.PC1, pca_df.PC2)
	plt.title('My PCA Graph')
	plt.xlabel('PC1 - {0}%'.format(per_var[0]))
	plt.ylabel('PC2 - {0}%'.format(per_var[1]))

	for sample in pca_df.index:
		plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

	plt.show()

	print(pc.components_[0])