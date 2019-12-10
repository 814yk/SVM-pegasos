#!/usr/bin/env python
# coding: utf-8

import sys
import time
from sklearn.datasets import load_svmlight_file
from scipy import sparse
import conjugateGradient as cg
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_grad,get_hessian,slack_variable,get_loss,get_acc



def main():
    # read the train file from first arugment
    train_file = sys.argv[1]
    #train_file='../data/covtype.scale.trn.libsvm'
    # read the test file from second argument
    test_file = sys.argv[2]
    #test_file = '../data/covtype.scale.tst.libsvm'

    # You can use load_svmlight_file to load data from train_file and test_file
    X_train, y_train = load_svmlight_file(train_file)
    X_test, y_test = load_svmlight_file(test_file)

    # You can use cg.ConjugateGradient(X, I, grad, lambda_)
    # Main entry point to the program
    X_train = sparse.hstack([X_train, np.ones((X_train.shape[0],1))])
    X_test = sparse.hstack([X_test, np.ones((X_test.shape[0],1))])

    X = sparse.csr_matrix(X_train)
    X_test=sparse.csr_matrix(X_test)

    y = sparse.csr_matrix(y_train).transpose()
    y_test = sparse.csr_matrix(y_test).transpose()


    #set global hyper parameter
    if sys.argv[1]=="covtype.scale.trn.libsvm" :
        lambda_ = 3631.3203125
        optimal_loss = 2541.664519
        five_fold_CV = 75.6661
        optimal_function_value = 2541.664519

    else:
        lambda_ = 7230.875
        optimal_loss = 669.664812
        five_fold_CV = 97.3655
        optimal_function_value = 669.664812



    #SGD
    #set local sgd hyper parameter
    print('starting SGD...')
    n_batch = 1000
    beta = 0
    lr = 0.001
    w = np.zeros((X_train.shape[1]))
    n = X_train.shape[0]
    sgd_grad = []
    sgd_time = []
    sgd_rel = []
    sgd_test_acc = []
    epoch=180
    start = time.time()
    #redefine learaning rate
    for i in range(epoch):
        gamma_t = lr / (1 + beta * i)
        batch_ = np.random.permutation(n) #shuffle
        for j in range(n//n_batch):
            #make batch
            idx=batch_[j*n_batch:(j+1)*n_batch]
            X_bc = X[idx]
            y_bc = y[idx]
            
            grad= get_grad(w, lambda_, n, X_bc, y_bc,n_batch) #comput gradient

            w = w - gamma_t * grad #update gradient

        t = time.time() - start
        sgd_time.append(t) # append to time list

        grad_ = np.linalg.norm(grad) # get gradient value
        sgd_grad.append(grad_)

        rel = (get_loss(w, lambda_,  X_test, y_test, n_batch) - optimal_loss) / optimal_loss # get relative func value
        sgd_rel.append(rel)
        
        test_acc = get_acc(w, lambda_, X_test, y_test, n_batch) # get test accuracy
        sgd_test_acc.append(test_acc)
    print("SGD : final_time: {}, fina_test_acc: {}".format(time.time() - start, sgd_test_acc[-1]))
    

    #plot SGD
    '''
    plt.plot(sgd_time, sgd_grad)
    plt.xlabel("time")
    plt.ylabel("grad")
    plt.title("SGD")
    plt.show()

    plt.plot(sgd_time, sgd_rel)
    plt.xlabel("time")
    plt.ylabel("relative function")
    plt.title("SGD")
    plt.show()


    plt.plot(sgd_time, sgd_test_acc)
    plt.xlabel("time")
    plt.ylabel("test_acc")
    plt.title("SGD")
    plt.show()

    '''
    print('starting Newton...')
    #Newton
    #set local newton hyper parameter
    epoch=50
    n_batch = 1000
    beta = 0.0001
    lr = 0.001
    w = np.zeros((X_train.shape[1]))
    n = X_train.shape[0]
    nt_grad = []
    nt_time = []
    nt_rel = []
    newton_time = time.time()

    nt_test_acc = []
    w = np.zeros((X_train.shape[1]))
    n = X_train.shape[0]
    

    for i in range(epoch):
        gamma_t = lr / (1 + beta * i)
        hessian_total = np.zeros(w.shape)
        I_ = [] #init I list to compute conjgate gradient
        for j in range(n//n_batch):
            X_bc = X[j*n_batch:(j+1)*n_batch] #make X_batch
            y_bc = y[j*n_batch:(j+1)*n_batch] #make y_batch

            hessian, I = get_hessian(w, lambda_, n, X_bc, y_bc) # get hessian
            hessian_total += hessian
            I_.append(I)
        I_ = np.concatenate(I_)
        hessian_total += w
        
        delta, _ = cg.conjugateGradient(X, I_, hessian_total, lambda_) #get update value from conjugateGradient

        w = w + delta #update w

        t = time.time() - newton_time
        nt_time.append(t) # append to time list
        
        grad_ = np.linalg.norm(hessian_total) # get gradient value
        nt_grad.append(grad_)

        rel = (get_loss(w, lambda_,  X_test, y_test, n_batch) - optimal_loss) / optimal_loss # get relative func value
        nt_rel.append(rel)

        test_acc = get_acc(w, lambda_, X_test, y_test, n_batch) # get test accuracy
        nt_test_acc.append(test_acc)
    final_time=time.time()-newton_time
    print("final_time: {}, fina_test_acc: {}".format(final_time, nt_test_acc[-1]))

    #plot
    '''
    plt.plot(nt_time, nt_grad)
    plt.xlabel("time")
    plt.ylabel("grad")
    plt.title("covtype Newton")
    plt.show()


    plt.plot(nt_time, nt_rel)
    plt.xlabel("time")
    plt.ylabel("relative function")
    plt.title("covtype Newton")
    plt.show()


    plt.plot(nt_time, nt_test_acc)
    plt.xlabel("time")
    plt.ylabel("test_acc")
    plt.title("covtype Newton")
    plt.show()

    '''



if __name__ == '__main__':
    main()
