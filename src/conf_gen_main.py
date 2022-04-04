
import conf_gen as cg # our conf gen library
import usr
import numpy as np
from rdkit import Chem
import csv
list1 = [] 
global j 
file = open('832usrfeatures.csv', 'w+', newline ='') 
suppl2 = Chem.SmilesMolSupplier("C:/Users/1901566.admin/Desktop/JPData/MUV/832d.smi")
suppl = Chem.SmilesMolSupplier("C:/Users/1901566.admin/Desktop/JPData/MUV/832a.smi")
j = 0 
decoys = 220   
actives = 25   #MUV has maximum 20- 30 actives for each target 
#We are using 220 inactives and 25 actives because some compounds may fail to generate conformers. In that case, we have 
#extra just in case in the end we end up with 200 inactives and 20 actives. 

ListOfDecoys = [] 
ListOfActives = []
a= []

for i in range(decoys):
    ListOfDecoys.append((Chem.MolToSmiles(suppl2[i])))   #to be converted back to SMILES for conformer generation
                        
                       
for mol in ListOfDecoys :
    m, e = cg.generate_conformers(mol)
    cg.save("low_e.sdf", m, e)
    try :
     a.append(usr.get_usr_descriptor(m))
     a[j].append(0)        #storing the class label in the list which will be helpful later for prediction
     j = j + 1 
    except  Exception as e: 
     continue 

a = a[:200] #all are inactives

for i in range(actives):
    ListOfActives.append(Chem.MolToSmiles(suppl[i]))
    
j = len(a)

for mol in ListOfActives :
    m, e = cg.generate_conformers(mol)
    cg.save("low_e.sdf", m, e)
    try :
     a.append(usr.get_usr_descriptor(m))
     a[j].append(1)
     j = j + 1       #storing the class label in the list which will be helpful later for prediction
    except  Exception as e: 
     continue 

a = a[:220]  #200 inactives, 20 actives


    

with file:        #storing the features of decoys and actives along with their labels in a csv file
    write = csv.writer(file) 
    write.writerows(a) 
