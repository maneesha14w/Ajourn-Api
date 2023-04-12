from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import string
import numpy as np
import tensorflow as tf
import pickle




app = Flask(__name__)
sent = SentimentIntensityAnalyzer()


def sent_analysis(text):
    scores = sent.polarity_scores(text)
    return scores


@app.route('/')
def index():
    print ('Hello, World! TensorFlow version: {}'.format(tf.__version__))
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

@app.route('/model', methods=['POST'])
def anxiety_detection():
    print('Request for anxiety detection received')
    text = request.get_json()['text']
    #text = ''.join(filter(lambda x: x in string.printable, text))
    text = "'{}'".format(text)
    print(text)

    sentiment_dict = sent_analysis(text)
    sent_arr = [value for value in sentiment_dict.values()]
    sent_arr.pop()
    sent_arr = np.array(sent_arr)
    sent_arr = np.reshape(sent_arr, (1, 3))
    
    # Load Tokenizer
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # #sequencing
    sequences = tokenizer.texts_to_sequences(text)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, truncating='post', padding='post', maxlen=3483) 

    # #concating sentiment
    inputs = np.append(padded, sent_arr)
    return jsonify(sequences)


if __name__ == '__main__':
   app.run()