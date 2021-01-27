from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def lasso(df, tgt,seed):
    X = df.drop(tgt, axis=1)

    y = df[tgt]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)

    try:
        l=Lasso(fit_intercept=True, normalize=False, copy_X=True).fit(X_train,y_train)
    except:
        l=Lasso(fit_intercept=True, normalize=False, copy_X=True).fit(X_train,y_train)

    predictions = l.predict(X_test)

    sns.distplot(((y_test-predictions)/y_test),bins=50)

    sc = l.score(X_test, y_test)

    print('Score:', sc)

    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

    X0 = pd.DataFrame([[37.6619, 0, 18.1, 0, 0.679, 6.202, 78.7, 1.8629, 24, 666, 20.2, 18.82, 14.52]],
                      columns=X_train.columns)  # 10.9
    X1 = pd.DataFrame([[13.0751, 0, 18.1, 0, 0.58, 5.713, 56.7, 2.8237, 24, 666, 20.2, 396.9, 14.76]],
                      columns=X_train.columns)  # 20.1
    X2 = pd.DataFrame([[0.06129, 20, 3.33, 1, 0.4429, 7.645, 49.7, 5.2119, 5, 216, 14.9, 377.07, 3.01]],
                      columns=X_train.columns)  # 46

    print("L: y0=", float(l.predict(X0)))
    print("L: y1=", float(l.predict(X1)))
    print("L: y2=", float(l.predict(X2)))

    return rmse