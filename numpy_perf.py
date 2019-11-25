#!/usr/bin/env python
# coding=utf-8

import numpy as np
import time

rows_a = 4000
cols_a = 3000
rows_b = cols_a
cols_b = 2000

a = np.random.randint(1024, size=(rows_a, cols_a)) 
b = np.random.randint(1024, size=(rows_b, cols_b))

start = time.time()
d = np.dot(a, b)
end = time.time()
print("integer time: {}".format(end - start))

a = np.random.rand(rows_a, cols_a)
b = np.random.rand(rows_b, cols_b)

start = time.time()
for _ in range(640):
  d = np.dot(a, b)
end = time.time()
print("double  time: {}".format(end - start))
