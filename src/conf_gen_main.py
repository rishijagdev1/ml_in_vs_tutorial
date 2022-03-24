
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
decoys = 300   
actives = 25   #MUV has maximum 20- 30 actives for each target 

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

for i in range(actives):
    ListOfActives.append(Chem.MolToSmiles(suppl[i]))
    

for mol in ListOfActives :
    m, e = cg.generate_conformers(mol)
    cg.save("low_e.sdf", m, e)
    try :
     a.append(usr.get_usr_descriptor(m))
     a[j].append(1)
     j = j + 1       #storing the class label in the list which will be helpful later for prediction
    except  Exception as e: 
     continue 


