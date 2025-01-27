# -*- coding: utf-8 -*-
import data
import model as ml
import tensorflow as tf
import sqlite3
import pandas as pd
import pickle
import numpy as np
from flask import g
from threading import Thread
from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter
import os
import predict
import math

# chatbot
# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-720116509362-733434984295-5uMX4w6HBxdwcGuX2k5sNASy"
SLACK_SIGNING_SECRET = "c2b024bde356c71f648e630affa1cd8b"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Req. 2-2-1 대답 예측 함수 구현
def predictHi(text):
    if __name__ == '__main__':
        return predict.predictOut(text)


# Req 2-2-2. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]
    text = " ".join(text.split()[1:])
    print(text)
    text = predictHi(text)
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

