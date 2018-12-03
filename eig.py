#!/usr/bin/python


import scipy.linalg as la
import numpy as np
from math import ceil, floor
#def hop(c):
#    A = np.mat([[1,1,6,1],[7,3,8,11],[9,5,1,10],[10,3.5,9,0.3]])
#    return np.dot(A, c)
#def hopH(c):
#    A = np.mat([[1,1,6,1],[7,3,8,11],[9,5,1,10],[10,3.5,9,0.3]])
#    return np.dot(A.H, c)
def msgbox(msg):
    msg_lines=msg.split('\n')
    max_length=max([len(line) for line in msg_lines])
    count = max_length +2 

    dash = "*"*count 
    print("*%s*" %dash)

    for line in msg_lines:
        dif=(max_length-len(line))
        if dif==0:
            print("* %s *"%line)
        else:
            print("* %s "%line+' '*dif+'*')    

    print("*%s*"%dash)
    print('\n')

def lanczos(hop, hopH, num_ci, iter_num):

    # A = np.mat([[1j,1,6j,1j],[7,3j,8,11j],[9,5,1j,10j],[10,3.5,9j,0.3j]]) 
    p = np.zeros((num_ci,1), dtype=complex)
    p = np.mat(p)
    p[0,0] = 1
    p[1,0] = 0.
    
    q = p
    
    r = np.mat(hop(q.A.T[0]).ravel()).T
    s = np.mat(hopH(p.A.T[0]).ravel()).T
    #print('THE FIRST ITERATION:\nr = \n{0}\ns = {1}'.format(r,s))
    ############## print('THE FIRST ITERATION') 
    a = np.dot(p.H, r)
    #print('a={0}'.format(a))
    r = np.mat(r - np.multiply(a, q))
    s = np.mat(s - np.multiply(a.H, p))
    #print('LOOP START: J=1.\nr={0},\ns={1}\na={2}'.format(r,s,a))
    ##############print('LOOP START: J=1')
    t = a
    
    w = r.H * s
    #print('w = {0}'.format(w))
    
    
    _b = np.sqrt(abs(w))
    _g = np.divide(w.H, _b)
    _q = np.divide(r, _b)
    _p = np.divide(s, _g.H)
    #print('_b = {0}\n_g={2}\n_q={2}\n_p={3}'.format(_b,_g,_q,_p)) 
    r = np.mat(hop(_q.A.T[0]).ravel()).T
    s = np.mat(hopH(_p.A.T[0]).ravel()).T
    r = r - np.multiply(_g, q)
    s = s - np.multiply(_b.H, p)
    
    if iter_num >= num_ci:
        iter_num = num_ci
    else:
        pass

    for i in range(2, iter_num+1):
        msgbox('@@@@@ THE {} ITERATION @@@@@'.format(i))
        q = _q
        p = _p
        b = _b
        g = _g
     
    #    print(q,b,p,g)
        a = p.H * r
        r = r - np.multiply(a, q)
        s = s - np.multiply(a.H, p)
    
        if i == 1:
            pass
        else:
            m = np.zeros((i-1,1),dtype=complex)
            m = np.mat(m)
            m[-1] = g
    
        t = np.c_[t,m]
        t = np.r_[t,np.zeros((1,i))]
        t[-1,-2] = b
        t[-1,-1] = a
        #######################msgbox('THE TRIDIAGONAL MATRIX IS\n{0}'.format(t))
        z, theta, omega = la.eig(t, left=True, right=True) # the eigentriplet (theta, z, omega) of T_j. T_j theta_ = theta z(left vector), omega^* T_j = z omega^*
        omega = np.mat(omega).H
        #######################msgbox('THE EIGENVALUES ARE:\n{0}'.format(z))
        #print(np.linalg.norm(r),np.linalg.norm(s))
        #if np.linalg.norm(r) ==0 or np.linalg.norm(s) ==0:
            
        #    break
        w = r.H * s
        #print(w)
        #if w==0:
        #    break
    
        _b = np.sqrt(abs(w))
        _g = np.divide(w.H, _b)
        _q = np.divide(r, _b)
        _p = np.divide(s, _g.H)
     
        r = np.mat(hop(_q.A.T[0]).ravel()).T
        s = np.mat(hopH(_p.A.T[0]).ravel()).T
        r = r - np.multiply(_g, q)
        s = s - np.multiply(np.conjugate(_b), p)
    return z, theta, omega
