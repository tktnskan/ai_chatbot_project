import pandas as pd
import numpy as np
import csv
import pickle
import tensorflow

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

"""
./advertising.csv 에서 데이터를 읽어, X와 Y를 만듭니다.

X는 (200, 3) 의 shape을 가진 2차원 np.array,
Y는 (200,) 의 shape을 가진 1차원 np.array여야 합니다.

X는 TV, Radio, Newspaper column 에 해당하는 데이터를 저장해야 합니다.
Y는 Sales column 에 해당하는 데이터를 저장해야 합니다.
"""

# Req 1-1-1. advertising.csv 데이터 읽고 저장
data = pd.read_csv('./advertising.csv')
X = pd.DataFrame(data=data, columns=["TV", "Radio", "Newspaper"])
print(data)
# A = data["Sales"]
# print(A.head())
Y = data.Sales
# print(Y.head())

# print(type(X))


# Req 1-1-2. 학습용 데이터와 테스트용 데이터로 분리합니다.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
# print(X_train)
# print(Y_train)
#
# print(X_train)
# print(X_test)

# """
# Req 1-2-1.
# LinearRegression()을 사용하여 학습합니다.
#
# 이후 학습된 beta값들을 학습된 모델에서 입력 받습니다.
#
# 참고 자료:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# """

lrmodel = LinearRegression()
lrmodel.fit(X_train, Y_train)
# print(lrmodel.coef_)
beta = lrmodel.coef_
# print(lrmodel.intercept_)
intercept = lrmodel.intercept_
# Req 1-2-2. 학습된 가중치 값 저장
beta_0 = intercept
beta_1 = beta[0]
beta_2 = beta[1]
beta_3 = beta[2]

print("Scikit-learn의 결과물")
print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("beta_2: %f" % beta_2)
print("beta_3: %f" % beta_3)
#
# # Req. 1-3-1.
# # X_test_pred에 테스트 데이터에 대한 예상 판매량을 모두 구하여 len(y_test) X 1 의 크기를 갖는 열벡터에 저장합니다.
X_test_pred = lrmodel.predict(X_test)
#
# """
# Mean squared error값을 출력합니다.
MSE = mean_squared_error(X_test_pred, Y_test)

# Variance score값을 출력합니다.

VarScore = r2_score(Y_test, X_test_pred)
#
# 함수를 찾아 사용하여 봅니다.
# https://scikit-learn.org/stable/index.html
# """
# # Req. 1-3-2. Mean squared error 계산
print("Mean squared error: %.2f" % MSE)
# # Req. 1-3-3. Variance score 계산
print("Variance score: %.2f" % VarScore)
#
# Req. 1-4-1.



def expected_sales(tv, rd, newspaper, beta_0, beta_1, beta_2, beta_3):

    ans = tv * beta_1 + rd * beta_2 + newspaper * beta_3 + beta_0

    return round(ans, 1)

# # Req. 1-4-2.
# # test 데이터에 있는 값을 직접적으로 넣어서 예상 판매량 값을 출력합니다.
# print(X_test.TV.values[1], Y_test.values[1])
print("TV: {}, Radio: {}, Newspaper: {} 판매량: {}".format(
   X_test.TV.values[0], X_test.Radio.values[0], X_test.Newspaper.values[0], Y_test.values[0]))
# #
print("예상 판매량: {}".format(expected_sales(
       float(X_test.TV.values[0]), float(X_test.Radio.values[0]), float(X_test.Newspaper.values[0]), beta_0, beta_1, beta_2, beta_3)))

#
# """
# Req. 1-5. pickle로 lrmodel 데이터 저장
# 파일명: model.clf
# """
with open('model.clf', 'wb') as f:
    pickle.dump(lrmodel, f, pickle.HIGHEST_PROTOCOL)

# with open('model.clf', 'rb') as f:
#     hi = pickle.load(f)
#
# print(hi)


# # Linear Regression Algorithm Part
# # 아래의 코드는 심화 과정이기에 사용하지 않는다면 주석 처리하고 실행합니다.
#
# """
# Req. 3-1-1.
# N_LinearRegression():
#
# Linear Regression 학습을 위한 알고리즘입니다.
# 학습데이터와 반복횟수를 받아서 최적의 직선(평면)으로 근사하는 가중치 값을 리턴합니다.
#
# 알고리즘 구성
# 1) 가중치 값인 beta_x, beta_3 초기화
# 2) Y label 데이터 reshape
# 3) 가중치 업데이트 과정 (iters번 반복)
# 3-1) prediction 함수를 사용하여 error 계산
# 3-2) gadient_beta 함수를 사용하여 가중치 값 업데이트
# 4) 가중치 값들 리턴
#
# """

def N_LinearRegression(X, Y, iters):
    """
    초기값 beta_0, beta_1, beta_2, beta_3 = 0
    여러가지 초기값을 실험해봅니다..
    초기값에 따라 iters간의 관계를 확인 가능합니다.
    """
    n_examples, n_features = np.shape(X)
    beta_x = np.zeros(n_features)
    beta_3 = 0

    #행렬 계산을 위하여 Y데이터의 사이즈를 (len(Y),1)로 저장합니다.
    # Y = np.array(Y)

    for i in range(iters):
        #실제 값 y와 예측 값(prediction()함수를 사용)의 차이를 계산하여 error를 정의합니다.
        error = prediction(X, beta_x, beta_3) - Y
        #gradient_beta함수를 통하여 델타값들을 업데이트 합니다.
        beta_x_delta, beta_3_delta = gradient_beta(X, error, learning_rate, n_examples)
        beta_x -= beta_x_delta
        beta_3 -= beta_3_delta

    return beta_x, beta_3

"""
Req. 3-1-2.
prediction():
beta값들을 받아서 예측값을 계산합니다.
X행렬의 크기와 beta의 행렬 크기를 맞추어 계산합니다.
"""
       
def prediction(X, beta_x, beta_3):
    # 예측 값을 계산하는 식을 만든다.

    equation = np.dot(X, beta_x) + beta_3

    return equation

"""
Req. 3-1-3.
gradient_beta():
beta값에 해당되는 gradient값을 계산하고 learning rate를 곱하여 출력합니다.
"""

def gradient_beta(X, error, lr, n):
    # beta_x를 업데이트하는 규칙을 정의한다.
    beta_x_delta = np.dot(X.transpose(), error) / n * lr
    # beta_3를 업데이트하는 규칙을 정의한다.
    beta_3_delta = np.mean(error) * lr

    return beta_x_delta, beta_3_delta


# N_LinearRegression 학습 파트

# Req 3-2-4. challenge
# 학습률(learning rate)를 설정합니다. (권장: 1e-3 ~ 1e-6)
learning_rate = 1e-6
# 반복 횟수(iteration)를 설정합니다. (자연수)
iteration = 3000

# Req. 3-2-1. 모델 학습
N_beta_x, N_beta_3 = N_LinearRegression(X_train.values, Y_train.values, iteration)

# Req. 3-2-2. 학습된 가중치 저장
print("\nN_LinearRegression의 결과물")
print("beta_0: %f" % N_beta_3)
print("beta_1: %f" % N_beta_x[0])
print("beta_2: %f" % N_beta_x[1])
print("beta_3: %f" % N_beta_x[2])


# Req. 3-3-1. 테스트 데이터의 예측 label값 계산
# X_test_pred에 테스트 데이터에 대한 예상 판매량을 모두 구하여 len(y_test) X 1 의 크기를 갖는 열벡터에 저장합니다.
N_X_test_pred = np.dot(X_test, N_beta_x)

# Req. 3-3-2. Mean squared error 계산
print("Mean squared error: %.2f" % mean_squared_error(Y_test, N_X_test_pred))
# Req. 3-3-3. Variance score 계산
print("Variance score: %.2f" % r2_score(Y_test, N_X_test_pred))

# Req. 3-4-1. 예상 판매량 출력
print("TV: {}, Radio: {}, Newspaper: {} 판매량: {}".format(
   X_test.TV.values[3], X_test.Radio.values[3], X_test.Newspaper.values[3], Y_test.values[3]))

print("예상 판매량: {}".format(N_X_test_pred[3]))


