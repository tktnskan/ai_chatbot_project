import numpy as np
import pickle
import os

import pandas as pd
from sklearn.metrics import accuracy_score
from konlpy.tag import Okt

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, GlobalMaxPooling1D, Dropout
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import KFold, train_test_split

# 데이터 셋 만들기, 맨 처음 한번만 실행
print (os.getcwd())

train_data = pd.read_csv("ratings_train.txt", sep="\t").fillna(" ")
test_data = pd.read_csv("ratings_test.txt", sep="\t").fillna(" ")

train_data.document = train_data.document.apply(lambda x: Okt().pos(x))
test_data.document = test_data.document.apply(lambda x: Okt().pos(x))

word_indices = {}

def make_indices(df, dictionary):
    for words in df.document.values:
        for word in words:
            if word[1] not in ["Josa", "Eomi"] and word not in dictionary.keys():
                dictionary[word] = len(dictionary)

make_indices(train_data, word_indices)
make_indices(test_data, word_indices)

with open("train_data.clf", "wb") as f:
    pickle.dump(train_data, f)
with open("test_data.clf", "wb") as f:
    pickle.dump(test_data, f)
with open("word_indices.clf", "wb") as f:
    pickle.dump(word_indices, f)

with open("./test_data.clf", "rb") as f:
    test_data = pickle.load(f)
    
with open("./train_data.clf", "rb") as f:
    train_data = pickle.load(f)
    
with open("./word_indices.clf", "rb") as f:
   word_indices = pickle.load(f)

train_data["sequences"] = train_data.document.apply(lambda x: [word_indices[xx] if xx in word_indices.keys() else None for xx in x])
test_data["sequences"] = test_data.document.apply(lambda x: [word_indices[xx] if xx in word_indices.keys() else None for xx in x])
train_data["sequences"] = train_data.sequences.apply(lambda x: [xx for xx in x if xx])
test_data["sequences"] = test_data.sequences.apply(lambda x: [xx for xx in x if xx])

test_data.head(5)

X_train_sequences = train_data.sequences.values
y_train = train_data.label.values
X_test_sequences = test_data.sequences.values
y_test = test_data.label.values

MAX_SEQ_LENGHT = len(max(train_data.sequences.values, key=len))
print("가장 형태소가 많은 문장의 경우", MAX_SEQ_LENGHT, "개")
# print(len(max(test_data.sequences.values, key=len))) 74개





N_FEATURES = len(word_indices)
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)
print(X_train_sequences)



X_train, X_test, Y_train, Y_test = train_test_split(X_train_sequences, y_train, test_size=0.2, random_state=7)

model = Sequential()
model.add(Embedding(len(word_indices) + 1, 256, input_length=MAX_SEQ_LENGHT))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

optimizer = Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

reducelr = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1)
modelcheck = ModelCheckpoint("sample.h5", save_best_only=True)
earlystop = EarlyStopping(patience=6)

model.fit(X_train, Y_train, epochs=500, batch_size= 40000, verbose=1, validation_data=(X_test, Y_test),  callbacks=[modelcheck, earlystop, reducelr])



model.load_weights("sample.h5")

preds = model.predict(X_test_sequences)
predicts = np.apply_along_axis(lambda x: 1 if x >= 0.5 else 0, 1, preds)

print(accuracy_score(predicts, y_test))  