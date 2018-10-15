import numpy as np

def rand():

    return np.random.random_sample()

def add(a, t, b):

    for i in range(len(a)):
        a[i] += t * b[i]

def get_dis(a,b):

    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def normalize(a, length):

    temp = get_dis(a,[0,0])
    a[0] *= length / temp
    a[1] *= length / temp

def get_fd(a, b, k):

    return [a[0] * k + b[0] * (1 - k),a[1] * k + b[1] * (1 - k)]