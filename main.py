# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def test():
    df = pd.read_csv("data.csv")
    print(df.head())

    X = df["설명변수"]
    y = df["목표변수"]
    plt.plot(X, y, 'o')

    line_fitter = LinearRegression()
    line_fitter.fit(X.values.reshape(-1, 1), y)

    print("기울기 : ", line_fitter.coef_)
    print("y절편 : ", line_fitter.intercept_)

    arr = [-11, 4, -9, 2, -10, -2, 0, -3, 1, -6, -7, -5, -8, -1]
    for i in arr:
        print(i, "->", line_fitter.predict([[i]]))

    plt.plot(X, y, 'o')
    plt.plot(X, line_fitter.predict(X.values.reshape(-1, 1)))
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
