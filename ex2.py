# -*- coding: utf-8 -*-
import time as tm
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp

time_start = tm.time()

n = [10, 50, 100, 200]

time_used1 = [0 for i in range(len(n))]
time_used2 = [0 for i in range(len(n))]
time_used3 = [0 for i in range(len(n))]
time_used4 = [0 for i in range(len(n))]
time_used5 = [0 for i in range(len(n))]
time_used6 = [0 for i in range(len(n))]
time_used7 = [0 for i in range(len(n))]
time_used8 = [0 for i in range(len(n))]

count = 1
iter_count = 50

err3 = [[0 for i in range(iter_count)] for j in range(len(n))]
err4 = [[0 for i in range(iter_count)] for j in range(len(n))]
err5 = [[0 for i in range(iter_count)] for j in range(len(n))]
err6 = [[0 for i in range(iter_count)] for j in range(len(n))]
err7 = [[0 for i in range(iter_count)] for j in range(len(n))]
err8 = [[0 for i in range(iter_count)] for j in range(len(n))]


for circle in range(count):
    for n_i in range(len(n)):
        
        # initialize the matrix
        A = np.random.rand(1, n[n_i])
        A = np.diag(A[0, :])
        U = np.random.rand(n[n_i], n[n_i])
        U = sp.linalg.orth(U)
        A = np.mat(U) * np.mat(A) * np.mat(U).T
        
        temp_A = A.copy()
        b = np.random.randn(n[n_i], 1)
        temp_b = b.copy()
        x = np.empty((1, n[n_i]))
        beg_x = np.random.randn(1, n[n_i])
        
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
        time_used1[n_i] = time_end - time_begin
        
        # Column Principle
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
        time_used2[n_i] = time_end - time_begin
        
        # Jacobi
        A = temp_A.copy()
        b = temp_b.copy()
        temp_x = beg_x.copy()
        next_x = np.empty((1, n[n_i]))
        time_begin = tm.time()
              
        for iter_c in range(iter_count):
            for i in range(n[n_i]):
                temp = b[i, 0]
                for j in range(n[n_i]):
                    if i != j:
                        temp -= A[i, j] * temp_x[0, j]
                next_x[0, i] = temp / A[i, i]
            temp_x = next_x.copy()
            
            # get the error
            err_sum = 0
            for i in range(n[n_i]):
                err_sum += math.fabs(temp_x[0, i] - x[0, i])
            err3[n_i][iter_c] = err_sum
        
        time_end = tm.time()
        time_used3[n_i] = time_end - time_begin
        
        
        # Gauss-Seidel
        A = temp_A.copy()
        b = temp_b.copy()
        time_begin = tm.time()
        temp_x = beg_x.copy()
        next_x = np.empty((1, n[n_i]))
        
        for iter_c in range(iter_count):
            for i in range(n[n_i]):
                temp = b[i, 0]
                for j in range(i):
                    temp -= A[i, j] * next_x[0, j]
                for j in range(i + 1, n[n_i]):
                    temp -= A[i, j] * temp_x[0, j]
                next_x[0, i] = temp / A[i, i]
            temp_x = next_x.copy()
            
            # get the error
            err_sum = 0
            for i in range(n[n_i]):
                err_sum += math.fabs(temp_x[0, i] - x[0, i])
            err4[n_i][iter_c] = err_sum
        
        time_end = tm.time()
        time_used4[n_i] = time_end - time_begin       


        # SOR - 0.5
        A = temp_A.copy()
        b = temp_b.copy()
        temp_x = beg_x.copy()
        next_x = np.empty((1, n[n_i]))
        omega = 0.5
        
        time_begin = tm.time()
        for iter_c in range(iter_count):
            for i in range(n[n_i]):
                temp = b[i, 0]
                for j in range(i):
                    temp -= A[i, j] * next_x[0, j]
                for j in range(i, n[n_i]):
                    temp -= A[i, j] * temp_x[0, j]
                next_x[0, i] = temp_x[0, i] + omega * temp / A[i, i]
            temp_x = next_x.copy()
            
            # get the error
            err_sum = 0
            for i in range(n[n_i]):
                err_sum += math.fabs(temp_x[0, i] - x[0, i])
            err5[n_i][iter_c] = err_sum
        
        time_end = tm.time()
        time_used5[n_i] = time_end - time_begin 


        # SOR - 1.5
        A = temp_A.copy()
        b = temp_b.copy()
        time_begin = tm.time()
        temp_x = beg_x.copy()
        next_x = np.empty((1, n[n_i]))
        omega = 1.5
        
        for iter_c in range(iter_count):
            for i in range(n[n_i]):
                temp = b[i, 0]
                for j in range(i):
                    temp -= A[i, j] * next_x[0, j]
                for j in range(i, n[n_i]):
                    temp -= A[i, j] * temp_x[0, j]
                next_x[0, i] = temp_x[0, i] + omega * temp / A[i, i]
            temp_x = next_x.copy()
            
            # get the error
            err_sum = 0
            for i in range(n[n_i]):
                err_sum += math.fabs(temp_x[0, i] - x[0, i])
            err7[n_i][iter_c] = err_sum
        
        time_end = tm.time()
        time_used7[n_i] = time_end - time_begin 


        # SOR - 1.8
        A = temp_A.copy()
        b = temp_b.copy()
        time_begin = tm.time()
        temp_x = beg_x.copy()
        next_x = np.empty((1, n[n_i]))
        omega = 1.8
        
        for iter_c in range(iter_count):
            for i in range(n[n_i]):
                temp = b[i, 0]
                for j in range(i):
                    temp -= A[i, j] * next_x[0, j]
                for j in range(i, n[n_i]):
                    temp -= A[i, j] * temp_x[0, j]
                next_x[0, i] = temp_x[0, i] + omega * temp / A[i, i]
            temp_x = next_x.copy()
            
            # get the error
            err_sum = 0
            for i in range(n[n_i]):
                err_sum += math.fabs(temp_x[0, i] - x[0, i])
            err8[n_i][iter_c] = err_sum
        
        time_end = tm.time()
        time_used8[n_i] = time_end - time_begin 


        # CG
        A = temp_A.copy()
        b = temp_b.copy()
        temp_x = beg_x.copy()
        next_x = np.empty((1, n[n_i]))
        time_begin = tm.time()

        # initialize r
        r = b - A.dot(temp_x.T)
        p = r.copy()
        
        for iter_c in range(iter_count):
            alpha = (r.T.dot(r) / (p.T.dot(A.dot(p))))[0, 0]
            
            temp_x = temp_x.T + alpha * p
            
            new_r = r - alpha * A.dot(p)
            belta = ((new_r.T.dot(new_r)) / (r.T.dot(r)))[0, 0]
            p = new_r + belta * p
            r = new_r.copy()
            
            # get the error
            err_sum = 0
            for i in range(n[n_i]):
                err_sum += math.fabs(temp_x.T[0, i] - x[0, i])
            err6[n_i][iter_c] = err_sum
        
        time_end = tm.time()
        time_used6[n_i] = time_end - time_begin 
     

# exercise 1
fig = plt.figure(1)
ax = plt.subplot(111)

ax.plot(n, time_used2, color = 'lightseagreen', label = 'Column Principle')
ax.plot(n, time_used1, color = 'salmon', label = 'Gauss Elimination')

plt.legend(loc='upper left')

plt.xticks(n, ["%d"%i for i in n], rotation=0)

plt.xlabel('Dimension')
plt.ylabel('Time')

plt.grid(linestyle="--", linewidth=0.5, color='black', alpha = 0.5)


# exercise 2 part2
fig = plt.figure(2)
ax = plt.subplot(111)

ax.plot(n, time_used2, color = 'lightseagreen', label = 'Column Principle')
ax.plot(n, time_used1, color = 'salmon', label = 'Gauss Elimination')
ax.plot(n, time_used3, color = 'pink', label = 'Jacobi')
ax.plot(n, time_used4, color = 'g', label = 'Gauss-Seidel')
ax.plot(n, time_used5, color = 'b', label = 'SOR - 0.5')
ax.plot(n, time_used7, color = 'y', label = 'SOR - 1.5')
ax.plot(n, time_used8, color = 'm', label = 'SOR - 1.8')
ax.plot(n, time_used6, color = 'r', label = 'CG')

plt.legend(loc='upper left')

plt.xticks(n, ["%d"%i for i in n], rotation=0)

plt.xlabel('Dimension')
plt.ylabel('Time')

plt.grid(linestyle="--", linewidth=0.5, color='black', alpha = 0.5)


# exercise 2 part1
fig = plt.figure(3)
ax = plt.subplot(111)

ax.plot([i for i in range(iter_count)], err3[0], color = 'pink', label = 'Jacobi')
ax.plot([i for i in range(iter_count)], err4[0], color = 'g', label = 'Gauss-Seidel')
ax.plot([i for i in range(iter_count)], err5[0], color = 'b', label = 'SOR - 0.5')
ax.plot([i for i in range(iter_count)], err7[0], color = 'y', label = 'SOR - 1.5')
ax.plot([i for i in range(iter_count)], err8[0], color = 'm', label = 'SOR - 1.8')
ax.plot([i for i in range(iter_count)], err6[0], color = 'r', label = 'CG')

plt.legend(loc='upper left')

plt.xlabel('Time')
plt.ylabel('Error')

plt.grid(linestyle="--", linewidth=0.5, color='black', alpha = 0.5)

fig = plt.figure(4)
ax = plt.subplot(111)

ax.plot([i for i in range(iter_count)], err3[1], color = 'pink', label = 'Jacobi')
ax.plot([i for i in range(iter_count)], err4[1], color = 'g', label = 'Gauss-Seidel')
ax.plot([i for i in range(iter_count)], err5[1], color = 'b', label = 'SOR - 0.5')
ax.plot([i for i in range(iter_count)], err7[1], color = 'y', label = 'SOR - 1.5')
ax.plot([i for i in range(iter_count)], err8[1], color = 'm', label = 'SOR - 2.0')
ax.plot([i for i in range(iter_count)], err6[1], color = 'r', label = 'CG')

plt.legend(loc='upper left')


plt.xlabel('Time')
plt.ylabel('Error')

plt.grid(linestyle="--", linewidth=0.5, color='black', alpha = 0.5)

fig = plt.figure(5)
ax = plt.subplot(111)

ax.plot([i for i in range(iter_count)], err3[2], color = 'pink', label = 'Jacobi')
ax.plot([i for i in range(iter_count)], err4[2], color = 'g', label = 'Gauss-Seidel')
ax.plot([i for i in range(iter_count)], err5[2], color = 'b', label = 'SOR - 0.5')
ax.plot([i for i in range(iter_count)], err7[2], color = 'y', label = 'SOR - 1.5')
ax.plot([i for i in range(iter_count)], err8[2], color = 'm', label = 'SOR - 2.0')
ax.plot([i for i in range(iter_count)], err6[2], color = 'r', label = 'CG')

plt.legend(loc='upper left')

plt.xlabel('Time')
plt.ylabel('Error')

plt.grid(linestyle="--", linewidth=0.5, color='black', alpha = 0.5)


fig = plt.figure(6)
ax = plt.subplot(111)

ax.plot([i for i in range(iter_count)], err3[3], color = 'pink', label = 'Jacobi')
ax.plot([i for i in range(iter_count)], err4[3], color = 'g', label = 'Gauss-Seidel')
ax.plot([i for i in range(iter_count)], err5[3], color = 'b', label = 'SOR - 0.5')
ax.plot([i for i in range(iter_count)], err7[3], color = 'y', label = 'SOR - 1.5')
ax.plot([i for i in range(iter_count)], err8[3], color = 'm', label = 'SOR - 2.0')
ax.plot([i for i in range(iter_count)], err6[3], color = 'r', label = 'CG')

plt.legend(loc='upper left')

plt.xlabel('Time')
plt.ylabel('Error')

plt.grid(linestyle="--", linewidth=0.5, color='black', alpha = 0.5)

plt.show()