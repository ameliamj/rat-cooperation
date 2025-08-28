#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 13:06:41 2025

@author: david
"""

import random
import matplotlib.pyplot as plt

def simulateRandom():
    domCount = 0
    subCount = 0
    for i in range(10):
        count1 = 0
        count2 = 0
        for i in range(15):
            number = random.choice([1, 2])
            
            if (number == 1):
                count1 += 1
            else:
                count2 += 1
        
        domCount += max(count1, count2)
        subCount += min(count1, count2)
    
    labels = ['Dominant', 'Submissive']
    sizes = [domCount, subCount]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.05, 0)  # Slightly separate the dominant slice

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=140, explode=explode, shadow=True)
    plt.title('First Lever Press Bias Across Sessions')
    plt.axis('equal')  # Ensures the pie is a circle
    plt.show()

simulateRandom()            
