# VAERS-Covid19
Project idea from Kaggle: COVID-19 World Vaccine Adverse Reactions by Ayush Garg.
Project for class CS584ML

## Note:
Raw data and merged dataset is not uploaded due to large size. Original data can be found: https://www.kaggle.com/ayushggarg/covid19-vaccine-adverse-reactions?select=2021VAERSVAX.csv

## License
CC0: Public Domain

## script.py
The script include five steps from data processing to graphing to training model and to using model for prediction.
### 1 Preprocess data 
- needs data files inside rawdata directory
- merge 3 datasets into 1
- preprocess text data into numbers, number to text mapping are saved under category directory
### 2 Visualize raw data 
- needs data files inside rawdata directory
- graph data into scatter plots 
### 3 Train model 
- needs merged dataset file
- split merged dataset into feature and target datasets
- Train model with 4 choices:
1. Stochastic Gradient Descent Classification
2. Logistic Regression Classification
3. K Neighbors Classification
4. Neural Network Classification with MLP
After training, model are saved under models directory.
### 4 Using model to predict patient outcomt
- needs trained and saved files inside models direcotory and input file
- print prediction category and predicted outcome for each patients in a list of 0 and 1.
