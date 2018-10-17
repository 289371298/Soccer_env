import numpy as np

def rand():

    return np.random.random_sample()

def add(a, t, b):

    for i in range(len(a)):
        a[i] += t * b[i]

def get_dis(a,b = [0,0]):

    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def normalize(a, length):

    temp = get_dis(a,[0,0])
    a[0] *= length / temp
    a[1] *= length / temp

def get_fd(a, b, k):

    return [a[0] * k + b[0] * (1 - k),a[1] * k + b[1] * (1 - k)]

def xmul(a, b, c):

        return (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])

def get_orth_point(aa, bb, cc):

        a = get_dis(aa, bb)
        b = get_dis(bb, cc)
        c = get_dis(aa, cc)
        if b < 0.01:
            return bb
        elif a < 0.01:
            return aa # 这里有bug
        else:
            #print('utils', a,b,c)
            cos_theta = ((a ** 2 + b ** 2 - c ** 2)/(2*a*b))
            len_orth = b * np.sqrt(1 - cos_theta ** 2)
            if xmul(bb, cc, aa) > 0:len_orth *= -1
            v_agent = bb - aa
            normalize(v_agent, 1)
            orth_point = cc + np.array([-v_agent[1], v_agent[0]]) * len_orth
            return orth_point
def mid(a, b, c):
    return (c >= a and c <= b) or (c <= a and c >= b)

def middle(a, b, c):

    return mid(a[0],b[0],c[0]) and mid(a[1],b[1],c[1])