#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import knn

train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [5, 5]])
query = np.array([6, 6])
kdt = knn.KDTree(train)
print(kdt.k_nearest(query, 3))
