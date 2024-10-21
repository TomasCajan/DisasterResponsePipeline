# Disaster Response Pipeline

#### Welcome to the Udacity Data Scientist Nanodegree capstone project II - a Disaster Response Pipeline.<br>
- This excercise is meant as a practice of writing ML pipelines - with focus on ETL, model training and deployment in a web-app.<br>
- The goal is to create a pipeline for classifying disaster-related messages using a machine learning classifier.<br>
- This project uses dataset shared Apen/Figure 8 company, containing real world messages sent during disasters.<br>

-----------

#### The Solution.<br>
- The solution is based on NLP-driven feature extraction using NLTK, serving as training data for a multioutput XGBoost classifier.<br>
- This classifier is trained to classify 36 different targets, being various types of disasters.<br>
- The predictions are linked to a flask-based web-app where user can see a visual output, along with graph representation of training dataset.<br>

-----------

#### How to run the project.<br>
Data needed : disasters_messages dataframe, disasters_categories dataframe <br>
**To run the ETL script, use command :**<br>
>python process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv ./data/disaster_database    <br>

**To run the ML Training script, use command :**<br>
>python train_classifier.py ./data/disaster_database.db ./models/classifier.pkl   <br>

**To launch the web-app :**<br>
>simply run the app.py script.<br>

-----------

###### Index screen view - visualization of training dataaset composition.<br>
![Project Diagram](screenshots/index.png)

###### Result screen view - classification of the inputed message.<br>
![Project Diagram](screenshots/result.png)
