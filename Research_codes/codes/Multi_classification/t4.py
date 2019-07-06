import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import time

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
#from textblob import Word
#from textblob import TextBlob
from nltk import PorterStemmer

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score,cohen_kappa_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from modAL.models import ActiveLearner,Committee
from modAL.uncertainty import uncertainty_sampling,margin_sampling,entropy_sampling
from modAL.disagreement import vote_entropy_sampling,max_disagreement_sampling,consensus_entropy_sampling


#import en_core_web_sm
#nlp = en_core_web_sm.load()

# %matplotlib inline
df = pd.read_pickle("../../dataset/final_dataset/sem_eval_train.pkl")
# df = df.fillna(0)
# df['cos_similarity'] = df['cos_similarity'].apply(lambda x: max(x,0))
# df['cos_similarity_demo'] = df['cos_similarity_demo'].apply(lambda x: max(x,0))


# df.fillna(-1, inplace=True)
# mms = MinMaxScaler()
# df[['length_ratio', 'aligned_score', 'aligned_score_demo', 'cos_similarity', 'cos_similarity_demo']] = \
# mms.fit_transform(df[['length_ratio', 'aligned_score', 'aligned_score_demo','cos_similarity', 'cos_similarity_demo']])
class Supervised_learner():
    def __init__(self,X,Y,model):
        self.X = X
        self.Y = Y
        self.model = model

    def learn(self):
        super_acc = []
        super_f1 = []
        super_kappa = []
        for i in np.linspace(0.05, 0.8, 20):
            X_train,X_test,Y_train,Y_test = train_test_split(self.X,self.Y,test_size = 1 - i)
            model = self.model
            model.fit(X_train, Y_train)
            model_accuracy = model.score(X_test, Y_test)
            super_acc.append(model_accuracy)
            model_pred = model.predict(X_test)
            model_f1 = f1_score(Y_test,model_pred,average ="weighted",labels=np.unique(model_pred))
            model_kappa = cohen_kappa_score(Y_test,model_pred)
            super_f1.append(model_f1)
            super_kappa.append(model_kappa)
        return super_acc, super_f1,super_kappa

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
        classes = self.short_df['grades_round'].unique()
        seed_index = []
        for i in classes:
            seed_index.append(self.short_df['grades_round'][self.short_df['grades_round']==i].index[0])
        seed_index

        act_data = self.short_df.copy()
        accuracy_list = []
        f1_total_list = []
        kappa_total_list = []

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
            model_pred = learner.predict(X_pool)
            f1_total_list.append(f1_score(y_pool,model_pred,average ="weighted",labels=np.unique(model_pred)))
            kappa_total_list.append(cohen_kappa_score(y_pool,model_pred))
        return accuracy_list,f1_total_list,kappa_total_list


class Active_learner_committee():
    def __init__(self,X,Y,data, percentage,query_method,learners):
        self.X = X
        self.Y = Y
        self.short_df = data.copy()
        self.percent = percentage
        self.query_method = query_method
        self.learners = learners
        
    def learn(self):       
        

        # seeding
        classes = self.short_df['grades_round'].unique()
        seed_index = []
        for i in classes:
            seed_index.append(self.short_df['grades_round'][self.short_df['grades_round']==i].index[0])
        seed_index

        act_data = self.short_df.copy()
        accuracy_list = []
        f1_total_list = []
        kappa_total_list = []

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
            learner_list= initiated_committee,
            query_strategy=vote_entropy_sampling
        )
        
        committee.teach(X_train,y_train)
        
        # pool-based sampling
        n_queries = int(len(X)/(100/self.percent))
        for idx in range(n_queries):
            query_idx, query_instance = committee.query(X_pool)   
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

            model_pred = committee.predict(X_pool)
            f1_total_list.append(f1_score(y_pool,model_pred,average ="weighted",labels=np.unique(model_pred)))
            kappa_total_list.append(cohen_kappa_score(y_pool,model_pred))
 
        return accuracy_list,f1_total_list,kappa_total_list     

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
        classes = self.short_df['grades_round'].unique()
        seed_index = []
        for i in classes:
            seed_index.append(self.short_df['grades_round'][self.short_df['grades_round']==i].index[0])
        seed_index

        act_data = self.short_df.copy()
        accuracy_list = []
        f1_total_list = []
        kappa_total_list = []

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

            model_pred = learner.predict(X_pool)
            f1_total_list.append(f1_score(y_pool,model_pred,average ="weighted",labels=np.unique(model_pred)))
            kappa_total_list.append(cohen_kappa_score(y_pool,model_pred))
        return accuracy_list,f1_total_list,kappa_total_list


class Random_learner_committee():
    def __init__(self,X,Y,data, percentage,learners):
        self.X = X
        self.Y = Y
        self.short_df = data.copy()
        self.percent = percentage
        self.learners = learners
        
    def learn(self):       
        # seeding
        classes = self.short_df['grades_round'].unique()
        seed_index = []
        for i in classes:
            seed_index.append(self.short_df['grades_round'][self.short_df['grades_round']==i].index[0])
        seed_index

        act_data = self.short_df.copy()
        accuracy_list = []
        f1_total_list = []
        kappa_total_list = []

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
            learner_list= initiated_committee,
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

            model_pred = committee.predict(X_pool)
            f1_total_list.append(f1_score(y_pool,model_pred,average ="weighted",labels=np.unique(model_pred)))
            kappa_total_list.append(cohen_kappa_score(y_pool,model_pred))
        return accuracy_list,f1_total_list ,kappa_total_list



### ================================Commitee based saving only commitee results===================================
## Active learner_commitee
X = df[['length_ratio','aligned_score','aligned_score_demo','cos_similarity','cos_similarity_demo']]
X = np.array(X) 
Y = df['grades_round'].values

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
query_methods = [vote_entropy_sampling,max_disagreement_sampling,consensus_entropy_sampling]
dict_accuracy_al = {}
dict_f1score_al = {}
dict_kappa_al = {}

with open('out4.txt', 'w+') as f:

for i,query_method in enumerate(query_methods):
        dict_accuracy_al[i] = []
        dict_f1score_al[i] = []
        dict_kappa_al[i] = []
        tic = time.time()
        ac_committee = Active_learner_committee(X,Y,df, Percent,query_method,models)
        accuracy_list,f1,kappa = ac_committee.learn()
        toc = time.time()
        print(query_method,(toc - tic)/(X.shape[0]*0.8))
        print(query_method,(toc - tic)/(X.shape[0]*0.8),file=f)
        dict_accuracy_al[i].append(accuracy_list)
        dict_f1score_al[i].append(f1)
        dict_kappa_al[i].append(kappa)