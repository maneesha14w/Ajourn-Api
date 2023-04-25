from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


import numpy as np
import tensorflow as tf
import pickle
import string


app = Flask(__name__)
sent = SentimentIntensityAnalyzer()
model = tf.keras.models.load_model('models/no_sent/weighted_model_without_sent_as_feature.h5')
# tf.keras.models.load_model('models/no sent/model_without_sent_as_feature.h5')
labels = {0: 'healthanxiety', 1: 'socialanxiety', 2: 'anxiety'}
tokenizer = 'models/no_sent/tokenizer_no_sent_weighted.pickle'
#tokenizer_no_sent = 'models/no sent/tokenizer_no_sent.pickle'

def sent_analysis(text):
    scores = sent.polarity_scores(text)
    return scores

def preprocessing(text, tokenizer):
    text = ''.join(filter(lambda x: x in string.printable, text))
    # Load Tokenizer
    with open(tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # sentiment_dict = sent_analysis(text)
    # sentiment_dict.pop('compound')

    #Sequencing
    sequences = tokenizer.texts_to_sequences(text)
  
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, truncating='post', padding='post', maxlen=3483)
  
    #input = np.concatenate((padded, sent_arr)) 
    return padded


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

@app.route('/model', methods=['POST'])
def anxiety_detection():
    print('Request for anxiety detection received')
    text = request.get_json()['text']
    print(text)
    
    input = preprocessing(text, tokenizer)
    
    pred = model.predict(input)
  
    # Find index of highest probability for the predicted class
    
    predicted_index = np.argmax(pred)
  
    print(predicted_index)
    
    # maps the prediction probabilities at the predicted_index position of the pred array to the corresponding label names in the labels dictionary.
    label_probs = dict(zip(labels.values(), pred[predicted_index]))
    
    print(label_probs)
    
    # Convert the dictionary values to float
    label_probs = {k: v.item() for k, v in label_probs.items()}
    label_probs = {k: round(v * 100, 1) for k, v in label_probs.items()}

    return jsonify(label_probs)


if __name__ == '__main__':
   app.run()



##