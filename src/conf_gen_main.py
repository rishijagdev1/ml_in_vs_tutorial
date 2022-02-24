import conf_gen as cg # our conf gen library
import usr
from rdkit import Chem
import csv
list1 = [] 

file = open('outa.csv', 'w+', newline ='') 
suppl = Chem.SmilesMolSupplier("C:/Users/1901566.admin/Desktop/JPData/MUV/600d.smi")


suppl = filter(None, suppl)
for mol in suppl:
    list1.append(Chem.MolToSmiles(mol))



'''list1 = list1[:25]''' 

a = []

for mol in list1 :
    m, e = cg.generate_conformers(mol)
    cg.save("low_e.sdf", m, e)
    a.append(usr.get_usr_descriptor(m))

    

with file:     
    write = csv.writer(file) 
    write.writerows(a) 
