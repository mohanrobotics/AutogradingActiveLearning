# Interactive-short-answer-grading

This repo contains report,codes python conda environment as an .yml file and GUI interface package.



Setting up the GUI:

npn install vue
sudo apt install nodejs-legacy

Steps for running the code:
The codes in this project can be run by setting up a python environment rnd_env.yml by running the following code in the terminal  directory.

conda env create -f rnd_env.yml
conda activate rnd_env


cd ./GUI/node-server/
node server.js

cd ./GUI/
conda activate rnd_env

python3 grader.py 

Note: place the autograded folder from NBgrader in the same directory where auto_grader.py is located. Also can change the directory in grader.py
where the autograded folder of nbgrader is located.


open google chrome
Turn on Allow-Control-Arrow-Origin(This plugin can be installed from https://chrome.google.com/webstore/detail/allow-control-allow-origi/nlfbmbojpeacfghkpbjhddihlkkiljbi)

Go to localhost:8080

Any issues please contact:
mohandass.psgrobo@gmail.com
kishaan96@gmail.com




Enjoy Grading!!!
