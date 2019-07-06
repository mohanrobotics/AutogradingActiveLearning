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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling,margin_sampling,entropy_sampling
from modAL.disagreement import vote_entropy_sampling,max_disagreement_sampling,consensus_entropy_sampling
import en_core_web_sm
import json
import os
import pickle as pkl

nlp = en_core_web_sm.load()

def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

from flask import Flask
from flask import jsonify
import json
from pprint import pprint
from flask import request, redirect
app = Flask(__name__)

threshold = 10
counter = 0
display_idx = 0

file_path = "./files/submitted/"


## getting all the names of the file
file_list = list()
for file in os.listdir(file_path):
    if file.endswith(".ipynb"):
        file_list.append(file)
file_list = sorted(file_list)

## craeting a dictionary of question anwers

ques_ans_dict = {q:[] for q in np.arange(0,17)}

for ff in file_list:
    with open(file_path+ff) as f:
        data = json.load(f)
    ques_count = 0
    ans_list = []
    ques_list = []
    for i in range(len(data['cells'])):
        if 'nbgrader' in data['cells'][i]['metadata'] and (data['cells'][i]['metadata']['nbgrader']['solution'])==True:
            ques_ans_dict[ques_count].append((data['cells'][i]['source']))
            ques_count+=1
            ques_list.append ((data['cells'][i-1]['source']))
            if ques_count == 17:
                break

## ordering and cleaning answers
answer_list = []
for i in range(len(ques_ans_dict.keys())):
    for answers in ques_ans_dict[i]:
        answer_list.append(''.join(answers))
## ordering and cleaning questions
ques_list = [i[2] for i in ques_list for _ in range(len(ques_ans_dict[0]))]

final_dict = {'question':ques_list, 'student_answer':answer_list ,\
             'question_id':[i for i in range(1,18) for _ in range(len(ques_ans_dict[0]))],\
            'student_id':[i.split(".")[0] + str(j) for j in range(1,18) for i in file_list]}
df = pd.DataFrame.from_dict(final_dict)

#====================================================
df_grades = pd.read_csv('./files/grades_17.csv')
df_grades.drop(columns='Question', inplace=True)
grades_list= []
for i in range(0,17):
    for items in list(df_grades.iloc[i].values):
        grades_list.append(items)
df['pre_grade'] = grades_list
#====================================================


#preprocessing
#converting to lower case
df['student_modified'] = df['student_answer'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df['student_modified'] = df['student_modified'].apply \
     (lambda x: " ".join(x for x in x.split() if x not in ['NO', '\n', 'YOUR', 'ANSWER', 'HERE', 'HERE:']))

#punctuation removal
df['student_modified'] = df['student_modified'].str.replace('[^\w\s]','')

#stop word removal
stop = stopwords.words('english')
df['student_modified'] = df['student_modified'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#lemmatisation
df['student_modified'] = df['student_modified'].apply(lambda x: " ".join([Word(word).lemmatize() for word in word_tokenize(x)]))


short_df = df[['student_answer','student_modified', 'pre_grade','question','question_id', 'student_id']]

short_df = short_df[:]

short_df["manual_grade"] = ["" for _ in range(len(short_df))]

# counting unique words in every student's answer
CV = CountVectorizer()
student_answer_count_vector = CV.fit_transform(short_df['student_modified'])
student_answer_count_vector = student_answer_count_vector.toarray()

X = student_answer_count_vector
Y = short_df['pre_grade'].values

#getting the seed index
classes = short_df['pre_grade'].unique()
seed_index = []
for i in classes:
	seed_index.append(short_df['pre_grade'][short_df['pre_grade']==i].index[0])

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
# lr = LogisticRegression()
learner = ActiveLearner(
	# estimator = lr,
    estimator = RandomForestClassifier(n_estimators=100),
#     estimator=KNeighborsClassifier(n_neighbors=3),
	X_training=X_train, y_training=y_train,
	query_strategy = margin_sampling
)




qn_ids = np.unique(short_df["question_id"].values)
qn_idx = 0
new_cells = []

@app.route('/get_question')
def get_question():
	print("get_question")


	global X_pool,y_pool,act_data,X,Y,learner,counter,threshold,display_idx,short_df

	cell = []
	ans_dict = {}

	ans_dict["is_last_qn"] = True
	if counter < threshold:
		query_idx, query_instance = learner.query(X_pool)

		ans_dict["question"] =  act_data.loc[int(query_idx),'question']
		ans_dict["answer"] =  act_data.loc[int(query_idx),'student_answer']
		ans_dict["auto_grade"] = "-"
		ans_dict["auto_grade_train_flag"]= True
		ans_dict["query_idx"] = int(query_idx)
		ans_dict["current"] = True
		if counter < threshold-1 :
			ans_dict["is_last_qn"] = False
		cell.append(ans_dict)
		print("Actual grade: ",y_pool[query_idx].reshape(1, ))

	return jsonify(cell)

@app.route('/display_data')
def display_data():
	global qn_ids,qn_idx,short_df
	if qn_idx <len(qn_ids):
		current_data = short_df[short_df["question_id"]== qn_ids[qn_idx]].to_dict('records')
		print (current_data[0].keys())
		final_data = []
		for ind_data in current_data:
			if ind_data["manual_grade"] == "":
				ind_data["manual_grade"] = ind_data["auto_grade"]
			ind_data["comment"] = []
			final_data.append(ind_data)
		qn_idx += 1
		return jsonify(final_data);
	else:
		return jsonify({1:0})


@app.route('/label_grade',methods = ['POST'])
def label_grade():

	print("inside_label_grade")

	global X_pool,y_pool,act_data,X,Y,learner,counter,threshold

	# accuracy_list.append(learner.score(X,Y))
	res = {}
	res['success'] = True
	counter += 1

	if counter < threshold :
		result = request.get_json(force=True)
		human_label = int(result['manual_grade'])
		query_idx = result['query_idx']

		learner.teach(X=X_pool[query_idx].reshape(1, -1),y=[y_pool[query_idx]])
		# remove queried instance from pool
		X_pool = np.delete(X_pool, query_idx, axis=0)
		y_pool = np.delete(y_pool, query_idx)

		act_data = act_data.drop(axis=0,index = query_idx)
		act_data.reset_index(drop=True, inplace=True)
		res['next_question'] = True

	else:
		print("inside counter is threshold")
		res['next_question'] = True
		res['accuracy'] = round(100.0*learner.score(X, Y),2)
		print(res['accuracy'])
		short_df['auto_grade']= pd.Series(learner.predict(X))

	return jsonify(res)



@app.route('/save_grade',methods = ['POST'])
def save_grade():
	global new_cells,qn_idx,qn_ids
	new_cell = request.get_json(force=True)
	new_cells += new_cell["saved_data"]
	res = {}
	res["saved_data"] = True
	if qn_idx == len(qn_ids):
		res["saved_data"] = False
		final_df = pd.DataFrame(new_cells)

		## writing the auto grade, manual grade and comment into an ipynb
		for student_id,ff in enumerate(file_list):
		    with open(file_path+ff) as f:
		        data = json.load(f)
		    answer_grade_count = student_id
		    for i in range(len(data["cells"])):
		        if ("nbgrader" in (data["cells"][i]["metadata"].keys())) and (answer_grade_count < len(final_df)):
		            if (data["cells"][i]["metadata"]["nbgrader"]["grade"]):
		                comment_dict = {}
		                comment_dict["cell_type"] = 'markdown'
		                comment_dict["metadata"] = {'deletable': False,'editable': False}
		                comment = ["<div class=\"alert alert-block alert-info\">\n",
		                    "\n",
		                    "### Auto-grade: {}\n".format(final_df["auto_grade"][answer_grade_count]),
		                    "\n",
		                    "### Manual grade: {}\n".format(final_df["manual_grade"][answer_grade_count]),
		                    "\n",
		                    "### Comment: {}\n".format(",".join(final_df["comment"][answer_grade_count])),
		                    "\n",
		                    "</div>"]
		                answer_grade_count+= len(final_df[final_df["question_id"]==1])
		                comment_dict["source"] = comment
		                data["cells"].insert(i+1,comment_dict)
		    if not os.path.exists(file_path+'../'+'/feedback/'):
		        os.makedirs(file_path+'../'+'/feedback/')
		    with open(file_path+'../'+'/feedback/'+ff.split(".")[0]+"_feedback.ipynb","w") as outfile:
		        json.dump(data,outfile)

	return jsonify(res)

if __name__ == "__main__":
	app.run(debug=True)
