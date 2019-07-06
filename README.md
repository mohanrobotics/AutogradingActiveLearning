# Interactive-short-answer-grading
Active learning is a machine learning paradigm which has seen useful applications in achieving good performance after learning from less amount of data. As supervised learning needs large amount of labeled data and short answer grading is not a "learn
once and apply forever" task, it would be a logical step to experiment active learning on this task. This project concentrates on evaluating the performance of different active learning strategies on the task of short answer grading on three different
datasets. In addition, useful features to be extracted from students’ answers and different machine learning models are studied. Performance of the best setting have been experimented and analyzed based on the metrics such as accuracy, f1-score, and run time. Furthermore, the efficiency of active learning with variable batch size and seeding have been studied. A web-based GUI is designed and implemented to incorporate a short answer grading system using the best active learning setting.
Based on the experiments conducted in this research, it is affirmed that active learning reaches the performance of supervised learning with less amount of graded answers for training. Selection of answers for grading while training the model based on active learning query strategy performed better than randomly sampled answers in all the datasets. Margin based uncertainty sampling was found to be efficient when compared to other query strategies in most of the experiments. State-of-the-art
features from Sultan et al., achieved better performance than bag of words and tf-idf features despite taking a long time to extract the features. The effectiveness of machine learning models was seen to be depended on the dataset used. Querying
the graded one by one and equal seeding worked well rather than batch querying.


This repository contains report,codes python conda environment as an .yml file and GUI interface package.
# Setting up the GUI:

'''
npn install vue
sudo apt install nodejs-legacy
'''

# Steps for running the code:
The codes in this project can be run by setting up a python environment rnd_env.yml by running the following code in the terminal  directory.

'''
conda env create -f rnd_env.yml
conda activate rnd_env
'''
'''
cd ./GUI/node-server/
node server.js
'''
'''
cd ./GUI/
conda activate rnd_env
python3 auto_grader.py 
'''

Note: place the autograded folder from NBgrader in the same directory where auto_grader.py is located. Also can change the directory in grader.py
where the autograded folder of nbgrader is located.


Open google chrome
Turn on Allow-Control-Arrow-Origin(This plugin can be installed from https://chrome.google.com/webstore/detail/allow-control-allow-origi/nlfbmbojpeacfghkpbjhddihlkkiljbi)

Go to localhost:8080

Any issues please contact:
mohandass.psgrobo@gmail.com
kishaan96@gmail.com




Enjoy Grading!!!
