from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import re
import string
import warnings
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

# Read Data
data = pd.read_csv("dataa.csv", delimiter=',', encoding='latin-1')

# Delet not needed columns
data.drop(['ids','date','flag','user'],axis = 1,inplace = True)

# Dividing data into positive and negative
positive_data = data[data.target==4].iloc[:,:]
negative_data = data[data.target==0].iloc[:,:]

# Cleaning the Tweets
data['Clean_Text'] = data['Text'].str.replace("@", "")
data['Clean_Text'] = data['Text'].str.replace(r"http\S+", "")
data['Clean_Text'] = data['Text'].str.replace("[^a-zA-Z]", " ")
data.head()

stopwords=nltk.corpus.stopwords.words('english')


def remove_stopwords(text):
    clean_text=' '.join([word for word in text.split() if word not in stopwords])
    return clean_text


data['Clean_Text'] = data['Clean_Text'].apply(lambda text : remove_stopwords(text.lower()))
data.head()

# Tokenization
data['Clean_Text'] = data['Clean_Text'].apply(lambda x: x.split())
data.head()

# Removing endings from words
from nltk.stem.porter import *
stemmer = PorterStemmer()
data['Clean_Text'] = data['Clean_Text'].apply(lambda x: [stemmer.stem(i) for i in x])
data.head()

# Joining the tokens together
data['Clean_Text'] = data['Clean_Text'].apply(lambda x: ' '.join([w for w in x]))
data.head()

data['Clean_Text'] = data['Clean_Text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
data.head()

count_vectorizer = CountVectorizer(stop_words='english')
cv = count_vectorizer.fit_transform(data['Clean_Text'])
cv.shape

X_train,X_test,y_train,y_test = train_test_split(cv,data['target'] , test_size=.2,stratify=data['target'], random_state=42)

# SVM Model
svc = svm.SVC()
svc.fit(X_train,y_train)
prediction_svc = svc.predict(X_test)
print(accuracy_score(prediction_svc,y_test))
print(confusion_matrix(y_test,prediction_svc))
print(classification_report(y_test,prediction_svc))

# KNN Model
KNN = KNeighborsClassifier(n_neighbors = 4)
KNN.fit(X_train, y_train)
prediction_knn = KNN.predict(X_test)
print(accuracy_score(prediction_knn,y_test))
print(confusion_matrix(y_test,prediction_knn))
print(classification_report(y_test,prediction_knn))
