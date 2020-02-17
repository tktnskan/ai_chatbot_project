import pickle
import pandas as pd
from threading import Thread
import sqlite3

import numpy as np
from konlpy.tag import Okt
from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from scipy.sparse import lil_matrix


# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-720116509362-733434984295-5uMX4w6HBxdwcGuX2k5sNASy"
SLACK_SIGNING_SECRET = "c2b024bde356c71f648e630affa1cd8b"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기

pickle_obj = open("train_Logistic_regression_model.clf", "rb")
word_indices = pickle.load(open("word_indices.clf", "rb"))
# print(word_indices)
# print(len(word_indices))
clf = pickle.load(pickle_obj)

# print(clf)

# Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리
def preprocess(input):
    okt = Okt()
    ans = okt.pos(input)
    X_test = lil_matrix((1, len(word_indices)))
    cnt = 0
    for row in ans:
        if row not in word_indices:
            print(row)
        else:
            X_test[cnt, word_indices[row]] = 1
    return X_test

# Req 2-2-3. 긍정 혹은 부정으로 분류
def classify(input):
    label = {0: '부정적 의견', 1: '긍정적 의견'}
    ans = clf.predict(input)
    print(ans)
    return label[ans[0]]

# Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장

def save_message(text):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("INSERT INTO search_history (query) VALUES (?)", (text,))
    conn.commit()
    conn.close()

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]
    text = " ".join(text.split()[1:])
    print(text)
    text = preprocess(text)
    save_message(text)
    text = classify(text)
    print(text)

    slack_web_client.chat_postMessage(
        channel=channel,
        text=text
    )

    return


@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"

if __name__ == '__main__':
    app.run()
