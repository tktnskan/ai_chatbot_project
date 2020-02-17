import pickle
import numpy as np

from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter


SLACK_TOKEN = "xoxb-720116509362-733434984295-5uMX4w6HBxdwcGuX2k5sNASy"
SLACK_SIGNING_SECRET = "c2b024bde356c71f648e630affa1cd8b"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req. 2-1-1. pickle로 저장된 model.clf 파일 불러오기

with open('model.clf', 'rb') as f:
    lrmodel = pickle.load(f)

# print(lrmodel)

# Req. 2-1-2. 입력 받은 광고비 데이터에 따른 예상 판매량을 출력하는 lin_pred() 함수 구현
def lin_pred(test_str):
    tv = test_str[0]
    radio = test_str[1]
    news = test_str[2]
    # print("222222222222222222222222222")
    # print(tv, radio, news)
    beta0 = lrmodel.intercept_
    beta1 = lrmodel.coef_[0]
    beta2 = lrmodel.coef_[1]
    beta3 = lrmodel.coef_[2]
    return str(round(tv * beta1 + radio * beta2 + news * beta3 + beta0, 1))

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    news = 0;
    tv = 0;
    radio = 0;
    cnt = 0
    # print(event_data)
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]
    if len(text.split()) > 4:
        send_text = text.split()[1:7]
        print(send_text)
        for i in send_text:
            print(i)
            if i == "신문" or i == "news" or i == "News" or i == "newspaper" or i == "Newpaper":
                print(cnt)
                print(send_text[cnt+1])
                news = int(send_text[cnt + 1])
                cnt += 1
            elif i == "티비" or i == "tv" or i == "Tv" or i == "TV" or i == "텔레비젼":
                print(cnt)
                print(send_text[cnt + 1])
                tv = int(send_text[cnt + 1])
                cnt += 1
            elif i == "라디오" or i == "radio" or i == "Radio" or i == "래디오":
                print(cnt)
                print(send_text[cnt + 1])
                radio = int(send_text[cnt + 1])
                cnt += 1
            else:
                cnt += 1

        real_text = [tv, radio, news]

    else:
        send_text = text.split()[1:4]
        real_text = [int(send_text[0]), int(send_text[1]), int(send_text[2])]


    # print('---------------------------------')
    # print(text.split()[1:4])
    keywords = "예상 판매량은 " + lin_pred(real_text) + " 입니다."
    # print(channel, keywords)
    slack_web_client.chat_postMessage(
        channel=channel,
        text=keywords
    )

    return

@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"

if __name__ == '__main__':
    app.run()
