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
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from modAL.models import ActiveLearner
from modAL.models import ActiveLearner
import en_core_web_sm
nlp = en_core_web_sm.load()

import gensim


def student_demoting(data):
	return " ".join(x for x in data['student_modified'].split() if x not in data['qn_modified'])

def ref_demoting(data):
	return " ".join(x for x in data['ref_modified'].split() if x not in data['qn_modified'])

model = gensim.models.KeyedVectors.load_word2vec_format('~/downloads/rnd/data/GoogleNews-vectors-negative300.bin.gz', binary=True)
def word_embed(word):
	try:
		vec = model[word]
		vec = vec.reshape(1,vec.shape[0])
	except:
		vec = np.ones((1, 300)) * 10e-3
		#this is hardcoded
	return vec

def model_embed_ref(data):
	sentence  = data ['ref_modified']
	sentence_array = [word_embed(word) for word in sentence.split()]
	return np.sum(sentence_array,axis=0)

def model_embed_stud(data):
        sentence  = data ['student_modified']
        sentence_array = [word_embed(word) for word in sentence.split()]
        return np.sum(sentence_array,axis=0)

original_data = pd.read_csv('~/downloads/rnd/data/mohler2_cleaned.csv')
original_data = original_data.drop(labels='Unnamed: 0', axis=1)
original_data = original_data.rename(columns={'question_number':'question_id','question_text':'question','answer_model':'ref_answer','answer_student':'student_answer','score_avg':'grade'})

df = original_data.copy()

#converting to lower case
df['qn_modified'] = df['question'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['ref_modified'] = df['ref_answer'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['student_modified'] = df['student_answer'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#punctuation removal
df['qn_modified'] = df['qn_modified'].str.replace('[^\w\s]','')
df['ref_modified'] = df['ref_modified'].str.replace('[^\w\s]','')
df['student_modified'] = df['student_modified'].str.replace('[^\w\s]','')

#stop word removal
#stop = stopwords.words('english')
#df['qn_modified'] = df['qn_modified'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#df['ref_modified'] = df['ref_modified'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#df['student_modified'] = df['student_modified'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


#lemmatisation
df['qn_modified'] = df['qn_modified'].apply(lambda x: " ".join([Word(word).lemmatize() for word in word_tokenize(x)]))
df['ref_modified'] = df['ref_modified'].apply(lambda x: " ".join([Word(word).lemmatize() for word in word_tokenize(x)]))
df['student_modified'] = df['student_modified'].apply(lambda x: " ".join([Word(word).lemmatize() for word in word_tokenize(x)]))

#question demoting
df['student_demoted'] = df.apply(student_demoting,axis=1)
df['ref_demoted'] = df.apply(ref_demoting,axis=1)

#length ratio
df['length_ratio'] = df['student_modified'].apply(lambda x: len(x)) / df['ref_modified'].apply(lambda x: len(x))

#rounding the grades
df['grades_round']= df['grade'].apply(lambda x: round(x))

#getting the word embeddings
df['embed_ref'] = df.apply(model_embed_ref,axis = 1)
df['embed_stud'] = df.apply(model_embed_stud,axis = 1)


# Bag of words creation
CV = CountVectorizer()
bag_of_word_vector = CV.fit_transform(df['student_modified'])
bag_of_word_vector = bag_of_word_vector.toarray()

# Tf-idf creation
Tf = TfidfVectorizer()
tfidf_vector = Tf.fit_transform(df['student_modified'])
tfidf_vector = tfidf_vector.toarray()

df.to_pickle("./feature.pkl")
