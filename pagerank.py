# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy.sparse import csr_matrix

f = open('soc-Epinions1.txt')

lines = f.readlines()

row = []
col = []
DELTA = 0.00000001
length = 75888
alpha = 0.8
count = [0] * length

for line in lines: 
    list = line.strip('\n').split('\t')
    row.append(int(list[0]))
    col.append(int(list[1]))
    count[int(list[0])] += 1

data = [1 / count[row[i]] for i in range(len(row))]
matrix = csr_matrix((data, (row, col)), shape=(length, length))
matrix = matrix.transpose()
row_p = [i for i in range(length)]
col_p = [0] * length
data_p = [1 / length] * length

poss = csr_matrix((data_p,(row_p, col_p)), shape=(length, 1))
count = 0


while True:
    
    count += 1
    new = alpha * matrix * poss + (1 - alpha) * poss
    
    check_stable = True
    for i in range(length):
       if math.fabs(poss[i, 0] - new[i, 0]) > DELTA:
           check_stable = False
           break
    poss = new
    print(count)
    if check_stable:
        break

max_i = 0
max_v = 0
for i in range(length):
    if poss[i, 0] > max_v:
        max_v = poss[i, 0]
        max_i = i

print(max_i)