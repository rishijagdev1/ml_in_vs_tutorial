from rdkit import Chem
from rdkit.Chem import AllChem

m = Chem.MolFromSmiles("c1ccccc1CNCC(=O)OC")
AllChem.EmbedMultipleConfs(m, 2)

w = Chem.SDWriter("test.sdf")
confs = m.GetConformers()
m.SetProp("name2", "test")
for c in confs:
    c.SetProp("name", "jp")
    w.write(m, confId=c.GetId())
w.close()