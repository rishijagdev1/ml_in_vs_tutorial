# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 02:45:26 2021

@author: 1901566.admin
"""

from sklearn import tree
import graphviz
from molvs import standardize
from molvs.standardize import Standardizer
import rdkit # compchem library
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import IPythonConsole 
from rdkit import rdBase


import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


header_list = ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8", "Feature9", "Feature10", "Feature11", "Feature12"]

data = pd.read_csv("C:/Users/1901566.admin/Desktop/JPData/DUD-E/GPCR/AA2AR/outd.csv", names = header_list) 

actives = 200
decoys = 200 


mol = pd.Series(decoys + actives) 
target_classes = np.array(['DECOY', 'ACTIVE'])
mol_labels = pd.Series(([target_classes[0]] * 200) + ([target_classes[1]] * 200))
data['class'] = mol_labels
data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75


train, test = data[data['is_train'] == True], data[data['is_train'] == False]

print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

features = data.columns[0:12]

y = pd.factorize(train['class'])


clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(train[features], y[0])



predictions = clf.predict(test[features])

clf.predict_proba(test[features])[:10]

preds = target_classes[predictions]


pd.crosstab(test['class'], preds, rownames=['Actual Class'], colnames=['Predicted Class'])


# accuracy
accuracy_score = metrics.accuracy_score(test['class'], preds)
print(accuracy_score)
# f1 score
f1_score = metrics.f1_score(test['class'], preds, pos_label="ACTIVE")
print(f1_score)



numeric_preds = [1 if cls == "ACTIVE" else 0 for cls in test['class']]

precision, recall, _ = precision_recall_curve(numeric_preds, predictions)

average_precision = average_precision_score(numeric_preds, predictions)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.05])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))



