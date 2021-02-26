#!/usr/bin/env python3

import conf_gen as cg # our conf gen library
import usr

m, e = cg.generate_conformers("CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C")
cg.save("low_e.sdf", m, e)
usr = usr.get_usr_descriptor(m)
print(usr)