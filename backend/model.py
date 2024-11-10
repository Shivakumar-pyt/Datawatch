import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Model:
    def __init__(self,path):
        self.df = pd.read_csv(path)
        self.data = pd.DataFrame(self.df)
        self.data = self.data.drop('Label',axis=1)
        self.model = None
    
    def run_kmeans(self,clusters):
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(self.data)
        labels = kmeans.predict(self.data)
        print(labels)
        return kmeans
    
    def train_logistic(self):

        X_train, X_test, y_train, y_test = train_test_split(self.data.drop("Label",axis=1), self.data["Label"], test_size=0.2, random_state=42)
    
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        self.model = clf
        return clf
    
    def test_kmeans(self,test_query):
        prediction = self.model.predict(test_query)
        return prediction
    
    def run_logistic(self,model,test_query):
        prediction = self.model.predict(test_query)
        return prediction
        

