from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("data.csv")
    print(df.head())  # 데이터 헤더

    x = df["설명변수"]
    y = df["목표변수"]

    line_fitter = LinearRegression()
    line_fitter.fit(x.values.reshape(-1, 1), y)  # 예측

    print("기울기 : ", line_fitter.coef_)
    print("y절편 : ", line_fitter.intercept_)

    res = pd.read_csv("result.csv")
    res['목표값'] = line_fitter.coef_ * res['새로운데이터'] + line_fitter.intercept_
    res.to_csv("result.csv", index=False)
    print(res.head())

    plt.plot(x, y, 'o')  # 데이터 그래프에 그리기
    plt.plot(x, line_fitter.predict(x.values.reshape(-1, 1)))  # 예측한 그래프 그리기
    plt.show()


if __name__ == '__main__':
    main()

