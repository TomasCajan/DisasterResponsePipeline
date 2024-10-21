# Disaster Response Pipeline

#### Welcome to the Udacity Data Scientist Nanodegree capstone project II - a Disaster Response Pipeline.<br>
- This excercise is meant as a practice of writing machine learning pipelines - with focus on ETL, model training and deployment in a web-app.<br>
- The main idea behind this project is to create a pipeline able to automatically identify disasters based on text messages around the social media using a machine learning classifier.<br>
- This project uses dataset shared Apen/Figure 8 company, containing real world messages sent during disasters.<br>

-----------

#### The Solution.<br>
- The core mechanism of this solution is NLP-based feature extraction using NLTK, serving as training data for a multioutput XGBoost classifier.<br>
- This classifier is trained to classify 36 different targets, being various types of disasters.<br>
- The predictions are linked to a flask-based web-app where user can see a visual output, along with graph representation of training dataset.<br>

###### Index screen view - visualization of training dataaset composition.<br>

