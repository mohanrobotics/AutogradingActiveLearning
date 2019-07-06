import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import Word
from textblob import TextBlob

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from modAL.models import ActiveLearner
from modAL.models import ActiveLearner

import en_core_web_sm
nlp = en_core_web_sm.load()

from flask import Flask
from flask import jsonify
import json
from pprint import pprint

app = Flask(__name__)
@app.route('/getdata')
def sample():

    # %matplotlib inline

    original_data = pd.read_csv('../dataset/mohler1_cleaned.csv')
    original_data = original_data.drop(labels='Unnamed: 6', axis=1)

    df = original_data.copy()

    #converting to lower case
    df['ref_modified'] = df['ref_answer'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df['student_modified'] = df['student_answer'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    #punctuation removal
    df['ref_modified'] = df['ref_modified'].str.replace('[^\w\s]','')
    df['student_modified'] = df['student_modified'].str.replace('[^\w\s]','')

    #stop word removal
    stop = stopwords.words('english')
    df['ref_modified'] = df['ref_modified'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df['student_modified'] = df['student_modified'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


    #lemmatisation
    df['ref_modified'] = df['ref_modified'].apply(lambda x: " ".join([Word(word).lemmatize() for word in word_tokenize(x)]))
    df['student_modified'] = df['student_modified'].apply(lambda x: " ".join([Word(word).lemmatize() for word in word_tokenize(x)]))

    short_df = df[['student_answer','student_modified', 'grade','question']]
    short_df['status'] = short_df['grade'] >= 3
    short_df['status'] = short_df['status'].astype(int)

    # short_df['word_count'] = short_df['student_answer'].apply(lambda x: dict(Counter(x.split())))

    # counting unique words in every student's answer
    CV = CountVectorizer()
    student_answer_count_vector = CV.fit_transform(short_df['student_modified'])
    student_answer_count_vector = student_answer_count_vector.toarray()

    X = student_answer_count_vector
    Y = short_df['status'].values

    #getting the seed index
    classes = short_df['status'].unique()
    seed_index = []
    for i in classes:
        seed_index.append(short_df['status'][short_df['status']==i].index[0])
        act_data = short_df.copy()
        accuracy_list = []

    # initialising
    # train_idx = [0, 5, 6 ,100]
    train_idx = seed_index
    X_train = X[train_idx]
    y_train = Y[train_idx]

    # generating the pool
    X_pool = np.delete(X, train_idx, axis=0)
    y_pool = np.delete(Y, train_idx)

    act_data = act_data.drop(axis=0,index = train_idx)
    act_data.reset_index(drop = True,inplace=True)


    # initializing the active learner
    lr = LogisticRegression()
    learner = ActiveLearner(
        estimator = lr,
    #     estimator = RandomForestClassifier(n_estimators=5),
    #     estimator=KNeighborsClassifier(n_neighbors=3),
        X_training=X_train, y_training=y_train
    )

    cell = []


    

    # pool-based sampling
    n_queries = 40
    for idx in range(n_queries):
        query_idx, query_instance = learner.query(X_pool)

        ans_dict = {}
        ans_dict["question"] = act_data.loc[int(query_idx),'question']
        ans_dict["answer"] = act_data.loc[int(query_idx),'student_answer']
        ans_dict["current"] = True;
        cell.append(ans_dict)

        return jsonify(cell)

        # print("\nQ: ", act_data.loc[int(query_idx),'question'])
        # print("A: ",act_data.loc[int(query_idx),'student_answer'])
        # print("Actual grade: ",y_pool[query_idx].reshape(1, ))
        # print ("Class probabilities: ",learner.predict_proba(X_pool[query_idx].reshape(1, -1)))
        # human_label = int(input("\nGive me a grade 0 or 1:"))
        learner.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=[human_label]
        )

        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

        act_data = act_data.drop(axis=0,index = query_idx)
        act_data.reset_index(drop=True, inplace=True)

        accuracy_list.append(learner.score(X,Y))
    #     print('Accuracy after query no. %d: %f' % (idx+1, learner.score(X, Y)))

if __name__ == "__main__":
    app.run()