"""
You should run this on an AWS instance unless you have a fast computer
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
import csv

def make_csv(name):
    with open(name, 'w') as csv_file:
        c1 = csv.writer(csv_file, delimiter=',')
        c1.writerow(['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10'])
        csv_file.close()

def write_to_csv(name, nparray):
    with open(name, 'a') as csv_file:
        c1 = csv.writer(csv_file, delimiter=',')
        for row in nparray:
            c1.writerow(row)
        csv_file.close()

completed=pd.read_csv('completed.csv')
completed['score']=completed['score'].replace(0,74)
dropped=pd.read_csv('dropped.csv')
dropped['score']=dropped['score'].replace(0,47)
watching=pd.read_csv('watching.csv')
watching=watching[watching['score']!=0]
on_hold=pd.read_csv('on_hold.csv')
on_hold=on_hold[on_hold['score']!=0]

table=pd.concat([completed, dropped, watching, on_hold])
del completed
del dropped
del watching
del on_hold
print 'table made'

table['username']=table['username']-1
table['anime_id']=table['anime_id']-1
matrix=csr_matrix((table['score'],(table['username'], table['anime_id'])))
del table
n_components=10
nmf=NMF(n_components=n_components, init='random', random_state=0)
nmf.fit(matrix)
print 'model fitted'

U = nmf.transform(matrix)
V = nmf.components_

make_csv('U.csv')
make_csv('V_T.csv')
write_to_csv('U.csv', U)
write_to_csv('V_T.csv', V.T)
