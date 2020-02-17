import numpy as np
import pickle
import pandas as pd

from sklearn.metrics import accuracy_score
from konlpy.tag import Okt
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt

"""
Req 1-1-1. 데이터 읽기
read_data(): 데이터를 읽어서 저장하는 함수
"""

def read_data(filename):
    data = pd.read_csv(filename, sep='\t').fillna("")
    return data

"""
Req 1-1-2. 토큰화 함수
tokenize(): 텍스트 데이터를 받아 KoNLPy의 okt 형태소 분석기로 토크나이징
"""

def tokenize(doc):
    okt = Okt()
    return okt.pos(doc)

"""
데이터 전 처리
"""

# train, test 데이터 읽기
# train_docs = pd.DataFrame(read_data('ratings_train.txt'))
# test_docs = pd.DataFrame(read_data('ratings_test.txt'))

# Req 1-1-2. 문장 데이터 토큰화
# train_docs, test_docs : 토큰화된 트레이닝, 테스트  문장에 label 정보를 추가한 list
# train_docs["token"] = tokenize(train_docs["document"])
# test_docs["token"] = tokenize(test_docs["document"])
# f = open("train_docs.clf", "wb")
# pickle.dump(train_docs, f)
# f = open("test_docs.clf", "wb")
# pickle.dump(test_docs, f)
#
f = open("train_docs.clf", "rb")
train_docs = pickle.load(f)

f = open("test_docs.clf", "rb")
test_docs = pickle.load(f)



# Req 1-1-3. word_indices 초기화
word_indices = {}

# Req 1-1-3. word_indices 채우기
index = 0
for row in train_docs["token"]:
    for i in row:
        if i not in word_indices:
            word_indices[i] = index
            index += 1

for row in test_docs["token"]:
    for i in row:
        if i not in word_indices:
            word_indices[i] = index
            index += 1

# print(word_indices)
f = open("word_indices.clf", "wb")
pickle.dump(word_indices, f)


# Req 1-1-4. sparse matrix 초기화
# X: train feature data
# X_test: test feature data
# print(word_indices)
X = lil_matrix((train_docs.shape[0], len(word_indices)))
X_test = lil_matrix((test_docs.shape[0], len(word_indices)))

#
print(X.shape)
print(X_test.shape)

# 평점 label 데이터가 저장될 Y 행렬 초기화
# Y: train data label
# Y_test: test data label
Y = train_docs['label'].values
Y_test = test_docs['label'].values
# print(Y)

# print(Y)
# print(Y_test)

# Req 1-1-5. one-hot 임베딩
# X,Y 벡터값 채우기
cnt = 0
for row in train_docs["token"]:
    # print(row)
    for i in row:
        X[cnt, word_indices[i]] = 1
    cnt += 1

cnt = 0
for row in test_docs["token"]:
    for i in row:
        X_test[cnt, word_indices[i]] = 1
    cnt += 1
# print(X)
# print(X_test)

"""
트레이닝 파트
clf  <- Naive baysian mdoel
clf2 <- Logistic regresion model
"""

# Req 1-2-1. Naive baysian mdoel 학습

clf = MultinomialNB()

# Req 1-2-2. Logistic regresion modoel 학습
clf2 = LogisticRegression(solver='sag', max_iter=10000)


clf.fit(X, Y)
# print("11111", X_test)
pred = clf.predict(X_test)
# print(pred)
clf2.fit(X, Y)
pred2 = clf2.predict(X_test)
# print(pred2)

"""
테스트 파트
"""

# Req 1-3-1. 문장 데이터에 따른 예측된 분류값 출력
print("Naive bayesian classifier example result: {}, {}".format(Y_test[3], pred[3]))
print("Logistic regression exampleresult: {}, {}".format(Y_test[3], pred2[3]))

# Req 1-3-2. 정확도 출력
print("Naive bayesian classifier accuracy: {}".format(accuracy_score(Y_test, pred)))
print("Logistic regression accuracy: {}".format(accuracy_score(Y_test, pred2)))

"""
데이터 저장 파트
"""

# Req 1-4. pickle로 학습된 모델 데이터 저장

f = open("train_Naive_baysian_model.clf", "wb")
pickle.dump(clf, f)
f = open("train_Logistic_regression_model.clf", "wb")
pickle.dump(clf2, f)



# Naive bayes classifier algorithm part
# 아래의 코드는 심화 과정이기에 사용하지 않는다면 주석 처리하고 실행합니다.

"""
Naive_Bayes_Classifier 알고리즘 클래스입니다.
"""

class Naive_Bayes_Classifier(object):

    """
    Req 3-1-1.
    log_likelihoods_naivebayes():
    feature 데이터를 받아 label(class)값에 해당되는 likelihood 값들을
    naive한 방식으로 구하고 그 값의 log값을 리턴
    """

    def log_likelihoods_naivebayes(self, feature_vector, Class):
        log_likelihood = 0.0

        if Class == 0:
            for feature_index in range(len(feature_vector)):
                if feature_vector[feature_index] == 1: #feature present
                    log_likelihood += None
                elif feature_vector[feature_index] == 0: #feature absent
                    log_likelihood += None
        elif Class == 1:
            for feature_index in range(len(feature_vector)):
                if feature_vector[feature_index] == 1:
                    log_likelihood += None
                elif feature_vector[feature_index] == 0:
                    log_likelihood += None

        return None

    """
    Req 3-1-2.
    class_posteriors():
    feature 데이터를 받아 label(class)값에 해당되는 posterior 값들을
    구하고 그 값의 log값을 리턴
    """

    def class_posteriors(self, feature_vector):
        log_likelihood_0 = self.log_likelihoods_naivebayes(feature_vector, Class = 0)
        log_likelihood_1 = self.log_likelihoods_naivebayes(feature_vector, Class = 1)

        log_posterior_0 = None
        log_posterior_1 = None

        return None

    """
    Req 3-1-3.
    classify():
    feature 데이터에 해당되는 posterir값들(class 개수)을 불러와 비교하여
    더 높은 확률을 갖는 class를 리턴
    """

    def classify(self, feature_vector):
        return None

    """
    Req 3-1-4.
    train():
    트레이닝 데이터를 받아 학습하는 함수
    학습 후, 각 class에 해당하는 prior값과 likelihood값을 업데이트

    알고리즘 구성
    1) 가중치 값인 beta_x_i, beta_c_i 초기화
    2) Y label 데이터 reshape
    3) 가중치 업데이트 과정 (iters번 반복)
    3-1) prediction 함수를 사용하여 error 계산
    3-2) gadient_beta 함수를 사용하여 가중치 값 업데이트
    4) 최적화 된 가중치 값들 리턴
       self.beta_x, self.beta_c
    """

    def train(self, X, Y):
        # label 0에 해당되는 데이터의 개수 값(num_0) 초기화
        num_0 = 0
        # label 1에 해당되는 데이터의 개수 값(num_1) 초기화
        num_1 = 0

        # Req 3-1-7. smoothing 조절
        # likelihood 확률이 0값을 갖는것을 피하기 위하여 smoothing 값 적용
        smoothing = None

        # label 0에 해당되는 각 feature 성분의 개수값(num_token_0) 초기화
        num_token_0 = np.zeros((1,X.shape[1]))
        # label 1에 해당되는 각 feature 성분의 개수값(num_token_1) 초기화
        num_token_1 = np.zeros((1,X.shape[1]))


        # 데이터의 num_0,num_1,num_token_0,num_token_1 값 계산
        for i in range(X.shape[0]):
            if (Y[i] == 0):
                num_0 += 1
                num_token_0 += None

            if (Y[i] == 1):
                num_1 += 1
                num_token_1 += None

        # smoothing을 사용하여 각 클래스에 해당되는 likelihood값 계산
        self.likelihoods_0 = None
        self.likelihoods_1 = None

        # 각 class의 prior를 계산
        prior_probability_0 = None
        prior_probability_1 = None

        # pior의 log값 계
        self.log_prior_0 = None
        self.log_prior_1 = None

        return None

    """
    Req 3-1-5.
    predict():
    테스트 데이터에 대해서 예측 label값을 출력해주는 함수
    """

    def predict(self, X_test):
        predictions = []
        X_test=X_test.toarray()
        if (len(X_test)==1):
            predictions.append(None)
        else:
            for case in X_test:
                predictions.append(None)

        return predictions

    """
    Req 3-1-6.
    score():
    테스트를 데이터를 받아 예측된 데이터(predict 함수)와
    테스트 데이터의 label값을 비교하여 정확도를 계산
    """

    def score(self, X_test, Y_test):

        return None

# Req 3-2-1. model에 Naive_Bayes_Classifier 클래스를 사용하여 학습합니다.
model = None

# Req 3-2-2. 정확도 측정
print("Naive_Bayes_Classifier accuracy: {}".format(None))

# Logistic regression algorithm part
# 아래의 코드는 심화 과정이기에 사용하지 않는다면 주석 처리하고 실행합니다.

"""
Logistic_Regression_Classifier 알고리즘 클래스입니다.
"""
class Logistic_Regression_Classifier(object):
    """
    Req 3-3-1.
    sigmoid():
    인풋값의 sigmoid 함수 값을 리턴
    """
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    """
    Req 3-3-2.
    prediction():
    X 데이터와 beta값들을 받아서 예측 확률P(class=1)을 계산.
    X 행렬의 크기와 beta의 행렬 크기를 맞추어 계산.
    ex) sigmoid(            X           x(행렬곱)       beta_x.T    +   beta_c)
                (데이터 수, feature 수)             (feature 수, 1)
    """

    def prediction(self, beta_x, beta_c, X):
        # 예측 확률 P(class=1)을 계산하는 식을 만든다.
        return self.sigmoid(X.dot(beta_x) + beta_c)

    """
    Req 3-3-3.
    gradient_beta():
    beta값에 해당되는 gradient값을 계산하고 learning rate를 곱하여 출력.
    """

    def gradient_beta(self, X, error, lr):
        # beta_x를 업데이트하는 규칙을 정의한다.
        beta_x_delta = X.T.dot(error) / (np.shape(X)[0]) * lr
        # beta_c를 업데이트하는 규칙을 정의한다.
        beta_c_delta = np.mean(error) * lr

        return beta_x_delta, beta_c_delta

    """
    Req 3-3-4.
    train():
    Logistic Regression 학습을 위한 함수.
    학습데이터를 받아서 최적의 sigmoid 함수으로 근사하는 가중치 값을 리턴.

    알고리즘 구성
    1) 가중치 값인 beta_x_i, beta_c_i 초기화
    2) Y label 데이터 reshape
    3) 가중치 업데이트 과정 (iters번 반복)
    3-1) prediction 함수를 사용하여 error 계산
    3-2) gadient_beta 함수를 사용하여 가중치 값 업데이트
    4) 최적화 된 가중치 값들 리턴
       self.beta_x, self.beta_c
    """

    def train(self, X, Y):
        # Req 3-3-8. learning rate 조절
        # 학습률(learning rate)를 설정한다.(권장: 1e-3 ~ 1e-6)
        lr = 1e-1
        # 반복 횟수(iteration)를 설정한다.(자연수)
        iters = 1000

        # beta_x, beta_c값을 업데이트 하기 위하여 beta_x_i, beta_c_i값을 초기화
        beta_x_i = np.zeros(X.shape[1])
        beta_c_i = 0

        #행렬 계산을 위하여 Y데이터의 사이즈를 (len(Y),1)로 저장합니다.
        # Y = np.array((len(X.shape[0]), 1))

        for i in range(iters):
            #실제 값 Y와 예측 값의 차이를 계산하여 error를 정의합니다.
            error = self.prediction(beta_x_i, beta_c_i, X) - Y
            #gredient_beta함수를 통하여 델타값들을 업데이트 합니다.
            beta_x_delta, beta_c_delta = self.gradient_beta(X, error, lr)
            beta_x_i -= beta_x_delta.T
            beta_c_i -= beta_c_delta
            print(i, beta_x_i, beta_c_i)

        self.beta_x = beta_x_i
        self.beta_c = beta_c_i

        print(self.beta_x, self.beta_c)


        return self.beta_x, self.beta_c

    """
    Req 3-3-5.
    classify():
    확률값을 0.5 기준으로 큰 값은 1, 작은 값은 0으로 리턴
    """

    def classify(self, X_test):
        X_test_predict = self.sigmoid(X_test.dot(self.beta_x) + self.beta_c)
        for idx in range(len(X_test_predict)):
            X_test_predict[idx] = 1 if X_test_predict[idx] > 0.5 else 0
        print(X_test_predict, X_test_predict.sum())
        return X_test_predict

    """
    Req 3-3-6.
    predict():
    테스트 데이터에 대해서 예측 label값을 출력해주는 함수
    """

    def predict(self, X_test):
        predictions = []
        X_test = X_test.toarray()
        if (len(X_test)==1):
            predictions.append(None)
        else:
            for case in X_test:
                predictions.append(None)

        return predictions


    """
    Req 3-3-7.
    score():
    테스트를 데이터를 받아 예측된 데이터(predict 함수)와
    테스트 데이터의 label값을 비교하여 정확도를 계산

    """
    def score(self, X_test, Y_test):
        X_test = self.classify(X_test)
        result = accuracy_score(X_test, Y_test)
        return result

# Req 3-4-1. model2에 Logistic_Regression_Classifier 클래스를 사용하여 학습합니다.

model2 = Logistic_Regression_Classifier()
model2.train(X, Y)
# model2.predict(X_test)

f = open('customLogistic', 'wb')
pickle.dump(model2, f)
f.close()

f = open('customLogistic', 'rb')
model2 = pickle.load(f)
f.close()

print(model2.score(X_test, Y_test))

# Req 3-4-2. 정확도 측정

print("Logistic_Regression_Classifier accuracy: {}".format(None))