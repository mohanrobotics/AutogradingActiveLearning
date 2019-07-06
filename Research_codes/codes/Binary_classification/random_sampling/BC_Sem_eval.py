
# coding: utf-8

# # Binary Classification Semeval

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import Word
from textblob import TextBlob
from nltk import PorterStemmer

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC


from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import uncertainty_sampling,margin_sampling,entropy_sampling

import en_core_web_sm
nlp = en_core_web_sm.load()

get_ipython().magic(u'matplotlib inline')


# In[2]:


df = pd.read_pickle("../../../dataset/final_dataset/sem_eval_train.pkl")

fig = plt.figure(figsize=(8,6))
df.groupby('status').student_modified.count().plot.bar(ylim=0)
plt.show()


# ## Active Learning

# In[3]:


class Active_learner():
    def __init__(self,X,Y,model,data, percentage,query_method):
        self.X = X
        self.Y = Y
        self.short_df = data.copy()
        self.percent = percentage
        self.model = model
        self.query_method = query_method
        
    def learn(self):       
        # seeding
        classes = self.short_df['status'].unique()
        seed_index = []
        for i in classes:
            seed_index.append(self.short_df['status'][self.short_df['status']==i].index[0])
        seed_index

        act_data = self.short_df.copy()
        accuracy_list = []

        # initialising
        train_idx = seed_index
        X_train = self.X[train_idx]
        y_train = self.Y[train_idx]

        # generating the pool
        X_pool = np.delete(self.X, train_idx, axis=0)
        y_pool = np.delete(self.Y, train_idx)

        act_data = act_data.drop(axis=0,index = train_idx)
        act_data.reset_index(drop = True,inplace=True)


        # initializing the active learner

        learner = ActiveLearner(
            estimator = self.model,
            X_training = X_train, y_training=y_train,
            query_strategy=self.query_method
        )

        # pool-based sampling
        n_queries = int(len(X)/(100/self.percent))
        for idx in range(n_queries):
            query_idx, query_instance = learner.query(X_pool)   
            learner.teach(
                X=X_pool[query_idx].reshape(1, -1),
                y=y_pool[query_idx].reshape(1, )
            )

            # remove queried instance from pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)

            act_data = act_data.drop(axis=0,index = query_idx)
            act_data.reset_index(drop=True, inplace=True)

            accuracy_list.append(learner.score(X_pool,y_pool))
#             print('Accuracy after query no. %d: %f' % (idx+1, learner.score(X_pool, y_pool)))
        print("By just labelling ",round(n_queries*100.0/len(X),2),"% of total data accuracy of ", round(learner.score(X_pool, y_pool),3), " % is achieved on the unseen data" )
        model_pred = learner.predict(X_pool)
        model_f1 = f1_score(y_pool,model_pred,average='weighted')
        return accuracy_list,model_f1


# ## Random sampling

# In[4]:


class Random_learner():
    def __init__(self,X,Y,model,data, percentage):
        self.X = X
        self.Y = Y
        self.short_df = data.copy()
        self.percent = percentage
        self.model = model
#         self.query_method = query_method
        
    def learn(self):       
        # seeding
        classes = self.short_df['status'].unique()
        seed_index = []
        for i in classes:
            seed_index.append(self.short_df['status'][self.short_df['status']==i].index[0])
        seed_index

        act_data = self.short_df.copy()
        accuracy_list = []

        # initialising
        train_idx = seed_index
        X_train = self.X[train_idx]
        y_train = self.Y[train_idx]

        # generating the pool
        X_pool = np.delete(self.X, train_idx, axis=0)
        y_pool = np.delete(self.Y, train_idx)

        act_data = act_data.drop(axis=0,index = train_idx)
        act_data.reset_index(drop = True,inplace=True)


        # initializing the random learner
        learner = ActiveLearner(
            estimator = self.model,
            X_training = X_train, y_training=y_train,
        )

        # pool-based sampling
        n_queries = int(len(X)/(100/self.percent))
        for idx in range(n_queries):
            query_idx = np.random.choice(range(len(X_pool)))   
            learner.teach(
                X=X_pool[query_idx].reshape(1, -1),
                y=y_pool[query_idx].reshape(1, )
            )

            # remove queried instance from pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)

            act_data = act_data.drop(axis=0,index = query_idx)
            act_data.reset_index(drop=True, inplace=True)

            accuracy_list.append(learner.score(X_pool,y_pool))
#             print('Accuracy after query no. %d: %f' % (idx+1, learner.score(X_pool, y_pool)))
        print("By just labelling ",round(n_queries*100.0/len(X),2),"% of total data accuracy of ", round(learner.score(X_pool, y_pool),3), " % is achieved on the unseen data" )
        model_pred = learner.predict(X_pool)
        model_f1 = f1_score(y_pool,model_pred,average='weighted')
        return accuracy_list,model_f1


# ## Random learning committee 

# In[5]:


class Random_learner_committee():
    def __init__(self,X,Y,data, percentage,learners):
        self.X = X
        self.Y = Y
        self.short_df = data.copy()
        self.percent = percentage
        self.learners = learners
        
    def learn(self):       
        # seeding
        classes = self.short_df['status'].unique()
        seed_index = []
        for i in classes:
            seed_index.append(self.short_df['status'][self.short_df['status']==i].index[0])
        seed_index

        act_data = self.short_df.copy()
        accuracy_list = []

        # initialising
        train_idx = seed_index
        X_train = self.X[train_idx]
        y_train = self.Y[train_idx]

        # generating the pool
        X_pool = np.delete(self.X, train_idx, axis=0)
        y_pool = np.delete(self.Y, train_idx)

        act_data = act_data.drop(axis=0,index = train_idx)
        act_data.reset_index(drop = True,inplace=True)
        
        initiated_committee = []
        for learner_idx,model in enumerate(self.learners):
            learner = ActiveLearner(
                estimator=model,
                X_training=X_train, y_training=y_train
            )
            initiated_committee.append(learner)
        
        # Commitee creation
        committee = Committee(
            learner_list = initiated_committee,
#             query_strategy=vote_entropy_sampling
        )
        
        committee.teach(X_train,y_train)
        
        # pool-based sampling
        n_queries = int(len(X)/(100/self.percent))
        for idx in range(n_queries):
            query_idx = np.random.choice(range(len(X_pool))) 
            committee.teach(
                X=X_pool[query_idx].reshape(1, -1),
                y=y_pool[query_idx].reshape(1, )
            )

            # remove queried instance from pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)

            act_data = act_data.drop(axis=0,index = query_idx)
            act_data.reset_index(drop=True, inplace=True)

            accuracy_list.append(accuracy_score(committee.predict(X_pool),y_pool))
#             print('Accuracy after query no. %d: %f' % (idx+1, accuracy_score(committee.predict(X_pool),y_pool)))
        print("By just labelling ",round(n_queries*100.0/len(X),2),"% of total data accuracy of ", round(accuracy_score(committee.predict(X_pool),y_pool),3), " % is achieved on the unseen data" )
        model_pred = committee.predict(X_pool)
        model_f1 = f1_score(y_pool,model_pred,average='weighted')
        return accuracy_list,model_f1        


# ## Sultan

# In[6]:


## Random learner
X = df[['length_ratio','aligned_score','aligned_score_demo','cos_similarity','cos_similarity_demo']]
X = np.array(X) 
Y = df['status'].values

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
dict_accuracy_al ={}
f1_score_list = {}
for i,model in enumerate(models):
    dict_accuracy_al[i] = []
    f1_score_list[i] = []
    print("******************************************************************************")
    rc = Random_learner(X,Y,model,df, Percent)
    accuracy_list,f1 = rc.learn()
    dict_accuracy_al[i].append(accuracy_list)
    f1_score_list[i].append(f1)
# dict_accuracy_al.to_pickle("../../results/dict_accuracy_al_mohler_"+str(Percent))
pkl.dump( dict_accuracy_al, open( "../../../results/random_sampling/Binary_class/dict_accuracy_random_semeval_"+str(Percent)+".pkl", "wb" ) )
pkl.dump( f1_score_list, open( "../../../results/random_sampling/Binary_class/f1_score_list_random_semeval_"+str(Percent)+".pkl", "wb" ) )


# In[7]:


## Random_committee learner
X = df[['length_ratio','aligned_score','aligned_score_demo','cos_similarity','cos_similarity_demo']]
X = np.array(X) 
Y = df['status'].values
Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
dict_accuracy_al = {}
f1_score_list= {}
print("******************************************************************************")
dict_accuracy_al[0] = []
rc_committee = Random_learner_committee(X,Y,df, Percent,models)
accuracy_list,f1 = rc_committee.learn()
dict_accuracy_al[0].append(accuracy_list)
f1_score_list[i].append(f1)
# dict_accuracy_al.to_pickle("../../results/dict_accuracy_al_mohler_"+str(Percent))
pkl.dump( dict_accuracy_al, open( "../../../results/random_sampling/Binary_class/dict_accuracy_random_com_semeval_"+str(Percent)+".pkl", "wb" ) )
pkl.dump( f1_score_list, open( "../../../results/random_sampling/Binary_class/f1_score_list_random_com_semeval_"+str(Percent)+".pkl", "wb" ) )


# ## Bag of Words

# In[8]:


#creating inputs and labels for BOW
CV = CountVectorizer()
student_answer_count_vector = CV.fit_transform(df['student_modified'])
student_answer_count_vector = student_answer_count_vector.toarray()

X = student_answer_count_vector
Y = df['status'].values

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
dict_accuracy_al ={}
f1_score_list = {}
for i,model in enumerate(models):
    dict_accuracy_al[i] = []
    f1_score_list[i] = []
    print("******************************************************************************")
    rc = Random_learner(X,Y,model,df, Percent)
    accuracy_list,f1 = rc.learn()
    dict_accuracy_al[i].append(accuracy_list)
    f1_score_list[i].append(f1)
# dict_accuracy_al.to_pickle("../../results/dict_accuracy_al_mohler_"+str(Percent))
pkl.dump( dict_accuracy_al, open( "../../../results/random_sampling/Binary_class/dict_accuracy_random_semeval_bag_"+str(Percent)+".pkl", "wb" ) )

pkl.dump( f1_score_list, open( "../../../results/random_sampling/Binary_class/f1_score_list_random_semeval_bag_"+str(Percent)+".pkl", "wb" ) )


# In[9]:


#creating inputs and labels for BOW
CV = CountVectorizer()
student_answer_count_vector = CV.fit_transform(df['student_modified'])
student_answer_count_vector = student_answer_count_vector.toarray()

X = student_answer_count_vector
Y = df['status'].values

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
dict_accuracy_al = {}
f1_score_list= {}
print("******************************************************************************")
dict_accuracy_al[0] = []
rc_committee = Random_learner_committee(X,Y,df, Percent,models)
accuracy_list,f1 = rc_committee.learn()
dict_accuracy_al[0].append(accuracy_list)
f1_score_list[i].append(f1)
# dict_accuracy_al.to_pickle("../../results/dict_accuracy_al_mohler_"+str(Percent))
pkl.dump( dict_accuracy_al, open( "../../../results/random_sampling/Binary_class/dict_accuracy_random_com_semeval_bag_"+str(Percent)+".pkl", "wb" ) )
pkl.dump( f1_score_list, open( "../../../results/random_sampling/Binary_class/f1_score_list_random_com_semeval_bag_"+str(Percent)+".pkl", "wb" ) )


# ## Tf-IDf

# In[10]:


#creating inputs and labels for TFidf
Tf = TfidfVectorizer()
student_answer_count_vector = Tf.fit_transform(df['student_modified'])
student_answer_count_vector = student_answer_count_vector.toarray()

X = student_answer_count_vector
Y = df['status'].values

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
dict_accuracy_al ={}
f1_score_list = {}
for i,model in enumerate(models):
    dict_accuracy_al[i] = []
    f1_score_list[i] = []
    print("******************************************************************************")
    rc = Random_learner(X,Y,model,df, Percent)
    accuracy_list,f1 = rc.learn()
    dict_accuracy_al[i].append(accuracy_list)
    f1_score_list[i].append(f1)

        
# dict_accuracy_al.to_pickle("../../results/dict_accuracy_al_mohler_"+str(Percent))
pkl.dump( dict_accuracy_al, open( "../../../results/random_sampling/Binary_class/dict_accuracy_random_semeval_tfidf_"+str(Percent)+".pkl", "wb" ) )

pkl.dump( f1_score_list, open( "../../../results/random_sampling/Binary_class/f1_score_list_random_semeval_tfidf_"+str(Percent)+".pkl", "wb" ) )


# In[ ]:



#creating inputs and labels for TFidf
Tf = TfidfVectorizer()
student_answer_count_vector = Tf.fit_transform(df['student_modified'])
student_answer_count_vector = student_answer_count_vector.toarray()

X = student_answer_count_vector
Y = df['status'].values

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
dict_accuracy_al = {}
f1_score_list= {}
print("******************************************************************************")
dict_accuracy_al[0] = []
rc_committee = Random_learner_committee(X,Y,df, Percent,models)
accuracy_list,f1 = rc_committee.learn()
dict_accuracy_al[0].append(accuracy_list)
f1_score_list[i].append(f1)
# dict_accuracy_al.to_pickle("../../results/dict_accuracy_al_mohler_"+str(Percent))
pkl.dump( dict_accuracy_al, open( "../../../results/random_sampling/Binary_class/dict_accuracy_random_com_semeval_tfidf_"+str(Percent)+".pkl", "wb" ) )
pkl.dump( f1_score_list, open( "../../../results/random_sampling/Binary_class/f1_score_list_random_com_semeval_tfidf_"+str(Percent)+".pkl", "wb" ) )

