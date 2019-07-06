import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import time

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


import en_core_web_sm
nlp = en_core_web_sm.load()

# %matplotlib inline
df = pd.read_pickle("../../dataset/final_dataset/mohler_final.pkl")
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

   
 ## Active learner
X = df[['length_ratio','aligned_score','aligned_score_demo','cos_similarity','cos_similarity_demo']]
X = np.array(X) 
Y = df['grades_round'].values

Percent = 80

models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators =100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
query_methods = [uncertainty_sampling,margin_sampling,entropy_sampling]
dict_accuracy_al ={}
dict_f1score_al = {}
dict_kappa_al = {}

for i,model in enumerate(models):
    dict_accuracy_al[i] = []
    dict_f1score_al[i] = []
    dict_kappa_al[i] = []

    for query_method in query_methods:
        ac = Active_learner(X,Y,model,df, Percent,query_method)
        tic = time.time()
        accuracy_list,f1,kappa = ac.learn()
        toc = time.time()
        print(str(model)[:15],query_method,(toc - tic)/(X.shape[0]*0.8))
        dict_accuracy_al[i].append(accuracy_list)
        dict_f1score_al[i].append(f1)
        dict_kappa_al[i].append(kappa)


# pkl.dump( dict_accuracy_al, open( "../../results/dict_accuracy_al_mohler_"+str(Percent)+".pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/dict_f1score_al_mohler_"+str(Percent)+".pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/dict_kappa_al_mohler_"+str(Percent)+".pkl", "wb" ) )


# supervised learning
X = df[['length_ratio','aligned_score','aligned_score_demo','cos_similarity','cos_similarity_demo']]
X = np.array(X) 
Y = df['grades_round'].values
iteration_count = 10

## Supervised learner
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators =100),SVC(kernel='linear' , probability=True),SVC(probability=True)]


dict_accuracy_sl_min = {}
dict_accuracy_sl_avg = {}
dict_accuracy_sl_max = {}

dict_f1score_sl_min = {}
dict_f1score_sl_avg = {}
dict_f1score_sl_max = {}

dict_kappa_sl_min = {}
dict_kappa_sl_avg = {}
dict_kappa_sl_max = {}

for i, model in enumerate(models):
    new_accuracy_list = []
    new_f1score_list = []
    new_kappa_list = []
    for _ in range(iteration_count):
        sl = Supervised_learner(X,Y,model)
        accuracy,f1score,kappa = sl.learn()
        new_accuracy_list.append(accuracy)
        new_f1score_list.append(f1score)
        new_kappa_list.append(kappa)
    dict_accuracy_sl_min[i]= np.min(new_accuracy_list,axis = 0)
    dict_accuracy_sl_avg[i]= np.mean(new_accuracy_list,axis = 0)
    dict_accuracy_sl_max[i]= np.max(new_accuracy_list,axis = 0)

    dict_f1score_sl_min[i]= np.min(new_f1score_list,axis = 0)
    dict_f1score_sl_avg[i]= np.mean(new_f1score_list,axis = 0)
    dict_f1score_sl_max[i]= np.max(new_f1score_list,axis = 0)

    dict_kappa_sl_min[i]= np.min(new_kappa_list,axis = 0)
    dict_kappa_sl_avg[i]= np.mean(new_kappa_list,axis = 0)
    dict_kappa_sl_max[i]= np.max(new_kappa_list,axis = 0)

# pkl.dump( dict_accuracy_sl_min, open( "../../results/dict_accuracy_min_mohler.pkl", "wb" ) )
# pkl.dump( dict_accuracy_sl_max, open( "../../results/dict_accuracy_max_mohler.pkl", "wb" ) )
# pkl.dump( dict_accuracy_sl_avg, open( "../../results/dict_accuracy_avg_mohler.pkl", "wb" ) )

# pkl.dump( dict_f1score_sl_min, open( "../../results/dict_f1score_min_mohler.pkl", "wb" ) )
# pkl.dump( dict_f1score_sl_max, open( "../../results/dict_f1score_max_mohler.pkl", "wb" ) )
# pkl.dump( dict_f1score_sl_avg, open( "../../results/dict_f1score_avg_mohler.pkl", "wb" ) )

# pkl.dump( dict_kappa_sl_min, open( "../../results/dict_kappa_min_mohler.pkl", "wb" ) )
# pkl.dump( dict_kappa_sl_max, open( "../../results/dict_kappa_max_mohler.pkl", "wb" ) )
# pkl.dump( dict_kappa_sl_avg, open( "../../results/dict_kappa_avg_mohler.pkl", "wb" ) )


#creating inputs and labels for BOW
CV = CountVectorizer()
student_answer_count_vector = CV.fit_transform(df['student_modified'])
student_answer_count_vector = student_answer_count_vector.toarray()

X = student_answer_count_vector
Y = df['grades_round'].values

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators =100)]
query_methods = [uncertainty_sampling,margin_sampling,entropy_sampling]
dict_accuracy_al ={}
dict_f1score_al = {}
dict_kappa_al = {}

for i,model in enumerate(models):
    dict_accuracy_al[i] = []
    dict_f1score_al[i] = []
    dict_kappa_al[i] = []

    for query_method in query_methods:
        ac = Active_learner(X,Y,model,df, Percent,query_method)
        tic = time.time()
        accuracy_list,f1,kappa = ac.learn()
        toc = time.time()
        print(str(model)[:15],query_method,(toc - tic)/(X.shape[0]*0.8))
        dict_accuracy_al[i].append(accuracy_list)
        dict_f1score_al[i].append(f1)
        dict_kappa_al[i].append(kappa)


# pkl.dump( dict_accuracy_al, open( "../../results/dict_accuracy_al_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/dict_f1score_al_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/dict_kappa_al_mohler_bag.pkl", "wb" ) )



# supervised learning

iteration_count = 10

## Supervised learner
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators =100)]


dict_accuracy_sl_min = {}
dict_accuracy_sl_avg = {}
dict_accuracy_sl_max = {}

dict_f1score_sl_min = {}
dict_f1score_sl_avg = {}
dict_f1score_sl_max = {}

dict_kappa_sl_min = {}
dict_kappa_sl_avg = {}
dict_kappa_sl_max = {}

for i, model in enumerate(models):
    new_accuracy_list = []
    new_f1score_list = []
    new_kappa_list = []
    for _ in range(iteration_count):
        sl = Supervised_learner(X,Y,model)
        accuracy,f1score,kappa = sl.learn()
        new_accuracy_list.append(accuracy)
        new_f1score_list.append(f1score)
        new_kappa_list.append(kappa)
    dict_accuracy_sl_min[i]= np.min(new_accuracy_list,axis = 0)
    dict_accuracy_sl_avg[i]= np.mean(new_accuracy_list,axis = 0)
    dict_accuracy_sl_max[i]= np.max(new_accuracy_list,axis = 0)

    dict_f1score_sl_min[i]= np.min(new_f1score_list,axis = 0)
    dict_f1score_sl_avg[i]= np.mean(new_f1score_list,axis = 0)
    dict_f1score_sl_max[i]= np.max(new_f1score_list,axis = 0)

    dict_kappa_sl_min[i]= np.min(new_kappa_list,axis = 0)
    dict_kappa_sl_avg[i]= np.mean(new_kappa_list,axis = 0)
    dict_kappa_sl_max[i]= np.max(new_kappa_list,axis = 0)

# pkl.dump( dict_accuracy_sl_min, open( "../../results/dict_accuracy_min_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_accuracy_sl_max, open( "../../results/dict_accuracy_max_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_accuracy_sl_avg, open( "../../results/dict_accuracy_avg_mohler_bag.pkl", "wb" ) )

# pkl.dump( dict_f1score_sl_min, open( "../../results/dict_f1score_min_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_f1score_sl_max, open( "../../results/dict_f1score_max_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_f1score_sl_avg, open( "../../results/dict_f1score_avg_mohler_bag.pkl", "wb" ) )

# pkl.dump( dict_kappa_sl_min, open( "../../results/dict_kappa_min_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_kappa_sl_max, open( "../../results/dict_kappa_max_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_kappa_sl_avg, open( "../../results/dict_kappa_avg_mohler_bag.pkl", "wb" ) )


#creating inputs and labels for TFidf
Tf = TfidfVectorizer()
student_answer_count_vector = Tf.fit_transform(df['student_modified'])
student_answer_count_vector = student_answer_count_vector.toarray()

X = student_answer_count_vector
Y = df['grades_round'].values

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators =100)]
query_methods = [uncertainty_sampling,margin_sampling,entropy_sampling]
dict_accuracy_al ={}
dict_f1score_al = {}
dict_kappa_al = {}

for i,model in enumerate(models):
    dict_accuracy_al[i] = []
    dict_f1score_al[i] = []
    dict_kappa_al[i] = []

    for query_method in query_methods:
        ac = Active_learner(X,Y,model,df, Percent,query_method)
        tic = time.time()
        accuracy_list,f1,kappa = ac.learn()
        toc = time.time()
        print(str(model)[:15],query_method,(toc - tic)/(X.shape[0]*0.8))
        dict_accuracy_al[i].append(accuracy_list)
        dict_f1score_al[i].append(f1)
        dict_kappa_al[i].append(kappa)


# pkl.dump( dict_accuracy_al, open( "../../results/dict_accuracy_al_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/dict_f1score_al_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/dict_kappa_al_mohler_tfidf.pkl", "wb" ) )
# supervised learning

iteration_count = 10

## Supervised learner
# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(), \
#           SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators =100)]


dict_accuracy_sl_min = {}
dict_accuracy_sl_avg = {}
dict_accuracy_sl_max = {}

dict_f1score_sl_min = {}
dict_f1score_sl_avg = {}
dict_f1score_sl_max = {}

dict_kappa_sl_min = {}
dict_kappa_sl_avg = {}
dict_kappa_sl_max = {}

for i, model in enumerate(models):
    new_accuracy_list = []
    new_f1score_list = []
    new_kappa_list = []
    for _ in range(iteration_count):
        sl = Supervised_learner(X,Y,model)
        accuracy,f1score,kappa = sl.learn()
        new_accuracy_list.append(accuracy)
        new_f1score_list.append(f1score)
        new_kappa_list.append(kappa)
    dict_accuracy_sl_min[i]= np.min(new_accuracy_list,axis = 0)
    dict_accuracy_sl_avg[i]= np.mean(new_accuracy_list,axis = 0)
    dict_accuracy_sl_max[i]= np.max(new_accuracy_list,axis = 0)

    dict_f1score_sl_min[i]= np.min(new_f1score_list,axis = 0)
    dict_f1score_sl_avg[i]= np.mean(new_f1score_list,axis = 0)
    dict_f1score_sl_max[i]= np.max(new_f1score_list,axis = 0)

    dict_kappa_sl_min[i]= np.min(new_kappa_list,axis = 0)
    dict_kappa_sl_avg[i]= np.mean(new_kappa_list,axis = 0)
    dict_kappa_sl_max[i]= np.max(new_kappa_list,axis = 0)

# pkl.dump( dict_accuracy_sl_min, open( "../../results/dict_accuracy_min_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_accuracy_sl_max, open( "../../results/dict_accuracy_max_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_accuracy_sl_avg, open( "../../results/dict_accuracy_avg_mohler_tfidf.pkl", "wb" ) )

# pkl.dump( dict_f1score_sl_min, open( "../../results/dict_f1score_min_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_f1score_sl_max, open( "../../results/dict_f1score_max_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_f1score_sl_avg, open( "../../results/dict_f1score_avg_mohler_tfidf.pkl", "wb" ) )

# pkl.dump( dict_kappa_sl_min, open( "../../results/dict_kappa_min_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_kappa_sl_max, open( "../../results/dict_kappa_max_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_kappa_sl_avg, open( "../../results/dict_kappa_avg_mohler_tfidf.pkl", "wb" ) )




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
for i,query_method in enumerate(query_methods):
    dict_accuracy_al[i] = []
    dict_f1score_al[i] = []
    dict_kappa_al[i] = []
    tic = time.time()
    ac_committee = Active_learner_committee(X,Y,df, Percent,query_method,models)
    toc = time.time()
    print(query_method,(toc - tic)/(X.shape[0]*0.8))
    accuracy_list,f1,kappa = ac_committee.learn()
    dict_accuracy_al[i].append(accuracy_list)
    dict_f1score_al[i].append(f1)
    dict_kappa_al[i].append(kappa)


# pkl.dump( dict_accuracy_al, open( "../../results/dict_accuracy_al_com_mohler_"+str(Percent)+".pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/dict_f1score_al_com_mohler_"+str(Percent)+".pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/dict_kappa_al_com_mohler_"+str(Percent)+".pkl", "wb" ) )


#creating inputs and labels for BOW
CV = CountVectorizer()
student_answer_count_vector = CV.fit_transform(df['student_modified'])
student_answer_count_vector = student_answer_count_vector.toarray()

X = student_answer_count_vector
Y = df['grades_round'].values
Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
query_methods = [vote_entropy_sampling,max_disagreement_sampling,consensus_entropy_sampling]
dict_accuracy_al = {}
dict_f1score_al = {}
dict_kappa_al = {}
for i,query_method in enumerate(query_methods):
    dict_accuracy_al[i] = []
    dict_f1score_al[i] = []
    dict_kappa_al[i] = []
    tic = time.time()
    ac_committee = Active_learner_committee(X,Y,df, Percent,query_method,models)
    toc = time.time()
    print(query_method,(toc - tic)/(X.shape[0]*0.8))
    accuracy_list,f1,kappa = ac_committee.learn()
    dict_accuracy_al[i].append(accuracy_list)
    dict_f1score_al[i].append(f1)
    dict_kappa_al[i].append(kappa)


# pkl.dump( dict_accuracy_al, open( "../../results/dict_accuracy_al_com_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/dict_f1score_al_com_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/dict_kappa_al_com_mohler_bag.pkl", "wb" ) )

#creating inputs and labels for TFidf
Tf = TfidfVectorizer()
student_answer_count_vector = Tf.fit_transform(df['student_modified'])
student_answer_count_vector = student_answer_count_vector.toarray()

X = student_answer_count_vector
Y = df['grades_round'].values

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
query_methods = [vote_entropy_sampling,max_disagreement_sampling,consensus_entropy_sampling]
dict_accuracy_al = {}
dict_f1score_al = {}
dict_kappa_al = {}
for i,query_method in enumerate(query_methods):
    dict_accuracy_al[i] = []
    dict_f1score_al[i] = []
    dict_kappa_al[i] = []
    tic = time.time()
    ac_committee = Active_learner_committee(X,Y,df, Percent,query_method,models)
    toc = time.time()
    print(query_method,(toc - tic)/(X.shape[0]*0.8))
    accuracy_list,f1,kappa = ac_committee.learn()
    dict_accuracy_al[i].append(accuracy_list)
    dict_f1score_al[i].append(f1)
    dict_kappa_al[i].append(kappa)


# pkl.dump( dict_accuracy_al, open( "../../results/dict_accuracy_al_com_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/dict_f1score_al_com_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/dict_kappa_al_com_mohler_tfidf.pkl", "wb" ) )



### Random Learner and random commitee learner ==============================================================



## Random learner
X = df[['length_ratio','aligned_score','aligned_score_demo','cos_similarity','cos_similarity_demo']]
X = np.array(X) 
Y = df['grades_round'].values

Percent = 80

models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
# query_methods = [uncertainty_sampling,margin_sampling,entropy_sampling]
dict_accuracy_al = {}
dict_f1score_al = {}
dict_kappa_al = {}
for i,model in enumerate(models):
    dict_accuracy_al[i] = []
    dict_f1score_al[i] = []
    dict_kappa_al[i] = []
#     for query_method in query_methods:
    rc = Random_learner(X,Y,model,df, Percent)
    accuracy_list,f1,kappa = rc.learn()
    dict_accuracy_al[i].append(accuracy_list)
    dict_f1score_al[i].append(f1)
    dict_kappa_al[i].append(kappa)


# pkl.dump( dict_accuracy_al, open( "../../results/random_sampling/Multi_class/dict_accuracy_random_mohler_"+str(Percent)+".pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/random_sampling/Multi_class/dict_f1score_random_mohler_"+str(Percent)+".pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/random_sampling/Multi_class/dict_kappa_random_mohler_"+str(Percent)+".pkl", "wb" ) )




## Random learner_commitee
X = df[['length_ratio','aligned_score','aligned_score_demo','cos_similarity','cos_similarity_demo']]
X = np.array(X) 
Y = df['grades_round'].values

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
dict_accuracy_al = {}
dict_f1score_al = {}
dict_kappa_al = {}
dict_accuracy_al[0] = []
dict_f1score_al[0] = []
dict_kappa_al[0] = []
rc_committee = Random_learner_committee(X, Y, df, Percent, models)
accuracy_list,f1,kappa = rc_committee.learn()
dict_accuracy_al[0].append(accuracy_list)
dict_f1score_al[0].append(f1)
dict_kappa_al[0].append(kappa)


# dict_accuracy_al.to_pickle("../../results/dict_accuracy_al_mohler_"+str(Percent))
# pkl.dump( dict_accuracy_al, open( "../../results/random_sampling/Multi_class/dict_accuracy_random_com_mohler_"+str(Percent)+".pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/random_sampling/Multi_class/dict_f1score_random_com_mohler_"+str(Percent)+".pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/random_sampling/Multi_class/dict_kappa_random_com_mohler_"+str(Percent)+".pkl", "wb" ) )


##BOW

CV = CountVectorizer()
student_answer_count_vector = CV.fit_transform(df['student_modified'])
student_answer_count_vector = student_answer_count_vector.toarray()

X = student_answer_count_vector
Y = df['grades_round'].values


Percent = 80

models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
# query_methods = [uncertainty_sampling,margin_sampling,entropy_sampling]
dict_accuracy_al = {}
dict_f1score_al = {}
dict_kappa_al = {}
for i,model in enumerate(models):
    dict_accuracy_al[i] = []
    dict_f1score_al[i] = []
    dict_kappa_al[i] = []
#     for query_method in query_methods:
    rc = Random_learner(X,Y,model,df, Percent)
    accuracy_list,f1,kappa = rc.learn()
    dict_accuracy_al[i].append(accuracy_list)
    dict_f1score_al[i].append(f1)
    dict_kappa_al[i].append(kappa)


# pkl.dump( dict_accuracy_al, open( "../../results/random_sampling/Multi_class/dict_accuracy_random_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/random_sampling/Multi_class/dict_f1score_random_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/random_sampling/Multi_class/dict_kappa_random_mohler_bag.pkl", "wb" ) )

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
dict_accuracy_al = {}
dict_f1score_al = {}
dict_kappa_al = {}
dict_accuracy_al[0] = []
dict_f1score_al[0] = []
dict_kappa_al[0] = []
rc_committee = Random_learner_committee(X, Y, df, Percent, models)
accuracy_list,f1,kappa = rc_committee.learn()
dict_accuracy_al[0].append(accuracy_list)
dict_f1score_al[0].append(f1)
dict_kappa_al[0].append(kappa)


# dict_accuracy_al.to_pickle("../../results/dict_accuracy_al_mohler_"+str(Percent))
# pkl.dump( dict_accuracy_al, open( "../../results/random_sampling/Multi_class/dict_accuracy_random_com_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/random_sampling/Multi_class/dict_f1score_random_com_mohler_bag.pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/random_sampling/Multi_class/dict_kappa_random_com_mohler_bag.pkl", "wb" ) )


## Tf - idf




#creating inputs and labels for TFidf
Tf = TfidfVectorizer()
student_answer_count_vector = Tf.fit_transform(df['student_modified'])
student_answer_count_vector = student_answer_count_vector.toarray()

X = student_answer_count_vector
Y = df['grades_round'].values

Percent = 80

models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
# query_methods = [uncertainty_sampling,margin_sampling,entropy_sampling]
dict_accuracy_al = {}
dict_f1score_al = {}
dict_kappa_al = {}
for i,model in enumerate(models):
    dict_accuracy_al[i] = []
    dict_f1score_al[i] = []
    dict_kappa_al[i] = []
#     for query_method in query_methods:
    rc = Random_learner(X,Y,model,df, Percent)
    accuracy_list,f1,kappa = rc.learn()
    dict_accuracy_al[i].append(accuracy_list)
    dict_f1score_al[i].append(f1)
    dict_kappa_al[i].append(kappa)


# pkl.dump( dict_accuracy_al, open( "../../results/random_sampling/Multi_class/dict_accuracy_random_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/random_sampling/Multi_class/dict_f1score_random_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/random_sampling/Multi_class/dict_kappa_random_mohler_tfidf.pkl", "wb" ) )

Percent = 80

# models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100),SVC(kernel='linear' , probability=True),SVC(probability=True)]
models = [LogisticRegression(),MultinomialNB(),RandomForestClassifier(n_estimators=100)]
dict_accuracy_al = {}
dict_f1score_al = {}
dict_kappa_al = {}
dict_accuracy_al[0] = []
dict_f1score_al[0] = []
dict_kappa_al[0] = []
rc_committee = Random_learner_committee(X, Y, df, Percent, models)
accuracy_list,f1,kappa = rc_committee.learn()
dict_accuracy_al[0].append(accuracy_list)
dict_f1score_al[0].append(f1)
dict_kappa_al[0].append(kappa)


# dict_accuracy_al.to_pickle("../../results/dict_accuracy_al_mohler_"+str(Percent))
# pkl.dump( dict_accuracy_al, open( "../../results/random_sampling/Multi_class/dict_accuracy_random_com_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_f1score_al, open( "../../results/random_sampling/Multi_class/dict_f1score_random_com_mohler_tfidf.pkl", "wb" ) )
# pkl.dump( dict_kappa_al, open( "../../results/random_sampling/Multi_class/dict_kappa_random_com_mohler_tfidf.pkl", "wb" ) )
