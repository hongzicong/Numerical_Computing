# -*- coding: utf-8 -*-
import time as tm
import numpy as np
import matplotlib.pyplot as plt

time_start = tm.time()

n = [10, 50, 100, 200]

time_used1 = [0 for i in range(len(n))]
time_used2 = [0 for i in range(len(n))]

count = 1

for circle in range(count):
    for n_i in range(len(n)):
        
        
        A = np.random.randn(n[n_i], n[n_i])
        temp_A = A.copy()
        b = np.random.randn(n[n_i], 1)
        temp_b = b.copy()
        x = np.empty((1, n[n_i]))
        
        
        # Gauss elimination
        time_begin = tm.time()
        for i in range(n[n_i] - 1):
            if A[i,i] == 0:
                for j in range(i + 1, n[n_i]):
                    if A[j,i] != 0:
                        A[[i, j], :] = A[[j, i], :]
                        b[[i, j], :] = b[[j, i], :]
                        break
            for j in range(i + 1, n[n_i]):
                
                
                m = A[j,i] / A[i,i]
                
                
                for k in range(i, n[n_i]):
                   A[j,k] -= m * A[i,k]
                   
                b[j,0] -= m * b[i,0]
        for i in range(n[n_i] - 1, -1, -1):
            for j in range(i + 1, n[n_i]):
                b[i, 0] -= A[i, j] * x[0, j]
            x[0, i] = b[i, 0] / A[i, i]
        time_end = tm.time()
        time_used1[n_i] += time_end - time_begin
        
        # column principle
        A = temp_A.copy()
        b = temp_b.copy()
        time_begin = tm.time()
        for i in range(n[n_i] - 1):
            
            max_i = i
            max_ele = A[i, i]
            for j in range(i + 1, n[n_i]):
                if A[j, i] > max_ele:
                    max_i = j
                    max_ele = A[j, i]
            if max_i != i:
                A[[i, max_i], :] = A[[max_i, i], :]
                b[[i, max_i], :] = b[[max_i, i], :]
            for j in range(i + 1, n[n_i]):
                m = A[j,i] / A[i,i]
                for k in range(i, n[n_i]):
                   A[j,k] -= m * A[i,k]
                   
                b[j,0] -= m * b[i,0]
        for i in range(n[n_i] - 1, -1, -1):
            for j in range(i + 1, n[n_i]):
                b[i, 0] -= A[i, j] * x[0, j]
            x[0, i] = b[i, 0] / A[i, i]
        time_end = tm.time()
        time_used2[n_i] += time_end - time_begin
        print(temp_A)
        print(A)
        print("--------")

for i in range(len(time_used1)):
    time_used1[i] /= count
    time_used2[i] /= count

fig = plt.figure(1)
ax = plt.subplot(111)

ax.plot(n, time_used2, color = 'lightseagreen', label = 'column principle')
ax.plot(n, time_used1, color = 'salmon', label = 'Gauss Elimination')
plt.plot()
plt.legend(loc='upper left')

plt.xticks(n, ["%d"%i for i in n], rotation=0)

plt.xlabel('Dimension')
plt.ylabel('Time')

plt.grid(linestyle="--", linewidth=0.5, color='black', alpha = 0.5)

plt.show()