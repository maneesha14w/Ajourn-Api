from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


import numpy as np
import tensorflow as tf
import pickle
import string


app = Flask(__name__)
sent = SentimentIntensityAnalyzer()
no_sent_model = tf.keras.models.load_model('models/no sent/model_without_sent_as_feature.h5')
labels = {0: 'healthanxiety', 1: 'socialanxiety', 2: 'anxiety'}
tokenizer_no_sent = 'models/no sent/tokenizer_no_sent.pickle'

def sent_analysis(text):
    scores = sent.polarity_scores(text)
    return scores

def preprocessing(text, tokenizer):
    print('2')
    text = ''.join(filter(lambda x: x in string.printable, text))
    print('3')
    # Load Tokenizer
    with open(tokenizer, 'rb') as handle:
        print('4')
        tokenizer = pickle.load(handle)

    # sentiment_dict = sent_analysis(text)
    # sent_arr = [value for value in sentiment_dict.values()]
    # sent_arr.pop()
    # print(sent_arr)
    print('5')
    #Sequencing
    sequences = tokenizer.texts_to_sequences(text)
    print('6')
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, truncating='post', padding='post', maxlen=3483)
    print('7')
    #input = np.concatenate((padded, sent_arr)) 
    return padded

@app.route('/pre',methods=['POST'])
# def pre():
#     text = request.get_json()['text']
#     padded = preprocessing(text)
#     sentiment_dict = sent_analysis(text)
#     print(sentiment_dict)
#     sent_arr = [value for value in sentiment_dict.values()]
#     sent_arr.pop()
#     sent_arr = np.array(sent_arr)
#     input = [padded, sent_arr]
#     input = np.array(input, dtype=object)
#     print(input)
#     print(input.shape)

#     input = input.tolist()

#     return jsonify()


@app.route('/')
def index():
    return 'Ajourn Api'


# TYPE OF REQUEST
# curl -X POST localhost:5000/sentiment -H "Content-type:application/json" -d "{\"text\": \"Does anyone else like taking long walks while it snows? Everything is quieter.\"}"
# VADER SENTIMENT METHOD

@app.route('/sent', methods=['POST'])
def sentiment_analysis():
    print('Request for sentiment received')
    text = request.get_json()['text']
    scores = sent.polarity_scores(text)
    return jsonify(scores)

@app.route('/model_no_sent', methods=['POST'])
def anxiety_detection():
    print('Request for anxiety detection received: MODEL NO SENT')
    text = request.get_json()['text']
    print(text)
    print('1')
    input = preprocessing(text, tokenizer_no_sent)
    print('8')
    pred = no_sent_model.predict(input)
    print('9')
    # Find index of highest probability for the predicted class
    print('10')
    predicted_index = np.argmax(pred)
    print('11')
    print(predicted_index)
    print('12')
    # maps the prediction probabilities at the predicted_index position of the pred array to the corresponding label names in the labels dictionary.
    label_probs = dict(zip(labels.values(), pred[predicted_index]))
    print('13')
    print(label_probs)
    print('14')
    # Convert the dictionary values to float
    label_probs = {k: v.item() for k, v in label_probs.items()}
    
    return jsonify(label_probs)


if __name__ == '__main__':
   app.run()



#    @app.route('/model', methods=['POST'])
# def anxiety_detection():
#     print('Request for anxiety detection received')
#     text = request.get_json()['text']
#     #text = ''.join(filter(lambda x: x in string.printable, text))
#     input = preprocessing(text)
    
#     text = "'{}'".format(text)
#     print(text)

#     sentiment_dict = sent_analysis(text)
#     sent_arr = [value for value in sentiment_dict.values()]
#     sent_arr.pop()
#     sent_arr = np.array(sent_arr)
#     sent_arr = np.reshape(sent_arr, (1, 3))
    
#     # Load Tokenizer
#     with open('tokenizer.pkl', 'rb') as handle:
#         tokenizer = pickle.load(handle)

#     # #sequencing
#     sequences = tokenizer.texts_to_sequences(text)
#     padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, truncating='post', padding='post', maxlen=3483) 

#     # #concating sentiment
#     inputs = np.append(padded, sent_arr)
#     return jsonify(sequences)
