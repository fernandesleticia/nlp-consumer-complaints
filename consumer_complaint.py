import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

''' Importing Dataset '''
df = pd.read_csv('data/complaints.csv')

''' Treating Data '''
#extracting columns
df = df[['Product', 'Consumer complaint narrative', 'Issue']]

#renaming Consumer complaint narrative column
df = df.rename(columns = {'Consumer complaint narrative': 'Narrative'})

# the fewer categories, the more accurate the result becomes
df = df.loc[(df['Product'] == 'Credit card')| (df['Product'] == 'Mortgage')| (df['Product'] == 'Student loan')| (df['Product'] == 'Consumer Loan')]

#removing nil values
df = df.loc[~df['Narrative'].isna()]
df = df.loc[~df['Issue'].isna()]

#creating NarrativeAndIssue concated column to achieve better results
df['NarrativeAndIssue'] = df['Issue'] +""+df['Narrative']

#reseting the index of the DataFrame using drop=True to avoid the old index being added as a column
df.reset_index(drop = True)

#removing duplicate rows for Product
df_aux = df['Product'].value_counts().reset_index()['index'].reset_index()

#renaming index column in place
df_aux.rename(columns = {'level_0' : 'Product_id', 'index' : 'Product'}, inplace = True)

#merging df and df_aux with inner join
df = df.merge(df_aux, on = 'Product', how= 'inner')

#deleting Product column
del df['Product']

''' Using smaller sample'''
df_sample = df.sample(5000).reset_index(drop = True)

''' Spliting Dataset into Train and Test '''
X = df_sample['NarrativeAndIssue'].to_list()
y = df_sample['Product_id'].to_list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=9)

''' Building KNN Classifier Model '''
#tokenize, and then performs a erm Frequency times Inverse Document Frequency transformation before passing the resulting features along to the classifier
pipe = Pipeline([
    ('counts', CountVectorizer()),
    ('tf_idf', TfidfTransformer(use_idf=True)),
    ('knn', KNeighborsClassifier())
])

knn_params = {
    'knn__n_neighbors':np.arange(1,11),
    'knn__metric': ['minkowski','cosine'],
    'knn__n_jobs': [-1]
}
#Exhaustive searching over specified parameter values for an estimator
knn_model = GridSearchCV(pipe, param_grid = knn_params, n_jobs = -1)

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)
knn_model.best_score_