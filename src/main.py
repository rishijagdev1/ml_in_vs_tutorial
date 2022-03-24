import rdkit # compchem library
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB


from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import IPythonConsole # use this to draw molecules in the notebook
from rdkit import rdBase
print(rdBase.rdkitVersion)

import pandas as pd
import numpy as np

suppla = []
suppld= []


from sklearn.ensemble import RandomForestClassifier

suppl = Chem.SmilesMolSupplier("C:/Users/1901566.admin/Desktop/JPData/MUV//832a.smi")
suppl1 =Chem.SmilesMolSupplier("C:/Users/1901566.admin/Desktop/JPData/MUV//832d.smi")

Num_Decoys = 200
Num_Actives = 20    #the ratio of actives to inactives is 1 :10

for i in range(Num_Actives):
    suppla.append(suppl[i])
    
for i in range(Num_Decoys):
    suppld.append(suppl1[i])    

decoys = suppld[:Num_Decoys]
actives = suppla[:Num_Actives]

mol = pd.Series(decoys + actives)
target_classes = np.array(['DECOY', 'ACTIVE'])
mol_labels = pd.Series(([target_classes[0]] * Num_Decoys) + ([target_classes[1]] *Num_Actives))
df = pd.DataFrame()
df['molecule'] = mol
df['class'] = mol_labels
df['mol_weight'] = [ Descriptors.MolWt(m) for m in df['molecule'] ]         #calculating molecular features and storing them in a database
df['rot_bonds'] =  [ Descriptors.NumRotatableBonds(m) for m in df['molecule'] ]
df['h_donors'] =   [ Descriptors.NumHDonors(m) for m in df['molecule'] ]
df['h_acceptors'] = [ Descriptors.NumHAcceptors(m) for m in df['molecule'] ]
df['log_p'] = [ Descriptors.MolLogP(m) for m in df['molecule'] ]
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75                   #separating training and test data


train, test = df[df['is_train'] == True], df[df['is_train'] == False]

print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))


features = df.columns[2:7]


y = pd.factorize(train['class'])

clf = RandomForestClassifier(n_estimators=100)

#clf = GaussianNB()


model = clf.fit(train[features], y[0])


predictions = clf.predict(test[features])

clf.predict_proba(test[features])

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
c_matrix = metrics.confusion_matrix(numeric_preds, predictions)
average_precision = average_precision_score(numeric_preds, predictions)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
