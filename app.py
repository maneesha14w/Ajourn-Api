from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

# import tensorflow as tf
#import numpy as np



app = Flask(__name__)
sent = SentimentIntensityAnalyzer()


def sentiment_analysis(text):
    scores = sent.polarity_scores(text)
    return scores

# def input_maker(text):
#     sent_arr = sentiment_analysis(text)
#     # Tokenize the input data
#     tokenizer = Tokenizer(num_words=1000, oov_token='<UNK>')
#     tokenizer.fit_on_texts(text)
#     #sequencing
#     sequences = tokenizer.texts_to_sequences(text)
#     padded = pad_sequences(sequences, truncating='post', padding='post', maxlen=3483) 
#     #concating sentiment
#     inputs = np.concatenate((padded, sent_arr), axis=1)
#     return inputs



@app.route('/')
def index():
    print('Request for index page received')
    return 'Hello World!'


# TYPE OF REQUEST
# curl -X POST localhost:5000/sentiment -H "Content-type:application/json" -d "{\"text\": \"Does anyone else like taking long walks while it snows? Everything is quieter.\"}"
# VADER SENTIMENT METHOD

@app.route('/sent', methods=['POST'])
def sentiment_analysis():
    print('Request for sentiment received')
    text = request.get_json()['text']
    scores = sent.polarity_scores(text)
    return jsonify(scores)

# @app.route('/model', methods=['POST'])
# def anxiety_detection():
#     print('Request for anxiety detection received')
#     text = request.get_json()['text']
#     input = input_maker(text)
#     return jsonify(input)


if __name__ == '__main__':
   app.run()