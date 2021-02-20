import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

df = pd.read_csv('data/complaints.csv')

df = df[['Product', 'Consumer complaint narrative']]

df.columns = ['Product', 'Narrative']

X = df['Narrative'].to_list()
y = df['Product'].to_list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=11)