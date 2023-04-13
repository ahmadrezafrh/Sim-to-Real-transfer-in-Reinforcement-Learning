#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:18:14 2023

@author: ahmadrezafrh
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

best_n_rows = 10
scores_dir = './dr_scores'
scores_name = 'domain_randomization_scores.csv'
x = []
y = []
z = []



scores = pd.read_csv(os.path.join(scores_dir, scores_name))
sorted_scores = pd.DataFrame(scores.sort_values('reward', ascending=False).to_numpy(), 
                             index=scores.index,
                             columns=scores.columns).drop(['Unnamed: 0'], axis=1)


for i in range(best_n_rows):
    print(f"the target reward: {sorted_scores['reward'][i]}")
    print(f"the distribution is: {sorted_scores['randomization distribution'][i]}\n")
    
rewards = sorted_scores['reward']
distributions = sorted_scores['randomization distribution']  
  
for i in range(sorted_scores.shape[0]):
    
    lower_bound = float(distributions[i][1:-1].split(', ')[0])
    upper_bound = float(distributions[i][1:-1].split(', ')[1])
    mean = (lower_bound + upper_bound)/2
    length = upper_bound - lower_bound
    reward = rewards[i]
    x.append(mean)
    y.append(length)
    z.append(reward)

x = np.array(x)
y = np.array(y)
z = np.array(z)

ax = plt.axes(projection='3d')
ax.set_xlabel('mean')
ax.set_ylabel('length')
ax.set_zlabel('reward');  
ax.scatter3D(x, y, z, c=z, cmap='Greens')
plt.show()

