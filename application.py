import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re, nltk # Sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")
@flask_app.route("/predict", methods = ["POST"])
def predict():
    def preparing(sentence):

        def clean(text):
            # Tokenizer
            import nltk
            from nltk.tokenize import sent_tokenize, word_tokenize

            def tokenizer_fct(sentence) :
                # print(sentence)
                sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
                word_tokens = word_tokenize(sentence_clean)
                return word_tokens

            # Stop words
            from nltk.corpus import stopwords
            stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']

            def stop_word_filter_fct(list_words) :
                filtered_w = [w for w in list_words if not w in stop_w]
                filtered_w2 = [w for w in filtered_w if len(w) > 2]
                return filtered_w2

            # lower case et alpha
            def lower_start_fct(list_words) :
                lw = [w.lower() for w in list_words if (not w.startswith("@")) 
        #                                   and (not w.startswith("#"))
                                            and (not w.startswith("http"))]
                return lw

            # Lemmatizer (base d'un mot)
            from nltk.stem import WordNetLemmatizer

            def lemma_fct(list_words) :
                lemmatizer = WordNetLemmatizer()
                lem_w = [lemmatizer.lemmatize(w) for w in list_words]
                return lem_w

            word_tokens = tokenizer_fct(text)
            sw = stop_word_filter_fct(word_tokens)
            lw = lower_start_fct(sw)
            lem_w = lemma_fct(lw)    
            transf_desc_text = ' '.join(lw)
            return transf_desc_text

        def vectorization(sentence):
            data = pd.read_csv("/home/souissi/Documents/Projet5/test/cleaned_data.csv")
            vectorizer=CountVectorizer(analyzer='word',min_df=2) # vectorizer: vector
            data_vectorized=vectorizer.fit_transform(data['text'])
            sentence = pd.Series(sentence)
            vectorized_data = vectorizer.transform(sentence)
            vectorized_data = pd.DataFrame(vectorized_data.toarray())
            return(vectorized_data)
    
        clean_text = clean(sentence)
        vector_text = vectorization(clean_text)
        return(vector_text)
    
    sentence = request.form["text"]
    sentence = preparing(sentence)
    
    prediction = model.predict(sentence)

    if prediction == 0:
        topic = 'android'
    elif prediction == 1:
        topic = 'c#'
    elif prediction == 2:
        topic = "c++"
    elif prediction == 3:
        topic = 'git'
    elif prediction == 4:
        topic = 'html'
    elif prediction == 5: 
        topic = 'ios'
    elif prediction == 6:
        topic = 'iphone'
    elif prediction == 7:
        topic = 'java'
    elif prediction == 8:
        topic = 'javascript'
    elif prediction == 9:
        topic = 'node.js'
    elif prediction == 10:
        topic = 'php'
    elif prediction == 11:
        topic = 'python'
    
    return render_template("index.html", prediction_text = "The Topic is {}".format(topic))

if __name__ == "__main__":
    flask_app.run(debug=True)