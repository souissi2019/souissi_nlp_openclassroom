import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import pickle

data = pd.read_csv("cleaned_data.csv")

#BOW
vectorizer=CountVectorizer(analyzer='word',min_df=2) # vectorizer: vector
data_vectorized=vectorizer.fit_transform(data['text'])

vectorized_data = pd.DataFrame(data_vectorized.toarray())

vectorized_data['tags'] = data['tags_filter']

data_1 = vectorized_data[(vectorized_data['tags'] == 'java') | (vectorized_data['tags'] == 'javascript') | (vectorized_data['tags'] == 'ios')
               | (vectorized_data['tags'] == 'android') | (vectorized_data['tags'] == 'c#') | (vectorized_data['tags'] == 'python') | 
               (vectorized_data['tags'] == 'php') | (vectorized_data['tags'] == 'git') | (vectorized_data['tags'] == 'node.js') | 
               (vectorized_data['tags'] == 'iphone') | (vectorized_data['tags'] == 'html') | (vectorized_data['tags'] == 'c++')]

le = preprocessing.LabelEncoder()
data_1['label'] = le.fit_transform(data_1.tags.values)

data_1 = data_1.drop(columns=['tags'],axis=1)
data_1 = data_1.reset_index()
X = data_1.drop(columns=['label'],axis=1)
X = X.drop(columns=['index'],axis=1)
y = data_1['label']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2)

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(C=3)
logisticRegr.fit(x_train, y_train)

# Make pickle file of our model
pickle.dump(logisticRegr, open("model.pkl", "wb"))