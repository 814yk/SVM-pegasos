import numpy as np

def get_grad(w, lambda_, n, X, y,n_batch):
    y_squeeze = np.squeeze(y.toarray())
    I = (1 - y_squeeze*X.dot(w)) > 0 #index
    X_I = X[I] #X_indexing
    y_I = y[I].toarray() #y_indexing
    y_I_squeeze = np.squeeze(y_I)
    grad = w / n_batch + 2 * lambda_ / n * X_I.transpose().dot((X_I.dot(w) - y_I_squeeze)) #compute_gradient
    return grad

def get_hessian(w, lambda_, n, X, y):
    y_squeeze = np.squeeze(y.toarray())
    I = (1 - y_squeeze*X.dot(w)) > 0 #index
    X_I = X[I] #X_indexing
    y_I = y[I] #y_indexing
    y_I_squeeze = np.squeeze(y_I.toarray())
    hessian = 2 * lambda_ / n * X_I.transpose().dot((X_I.dot(w) - y_I_squeeze)) #compute_hessian
    hessian=np.array(hessian,dtype=float)
    return hessian, I


def slack_variable(w, X, y):
    y_squeeze = np.squeeze(y.toarray())
    I = (1 - y_squeeze*X.dot(w)) > 0 #index
    X_I = X[I] #X_indexing
    y_I = y[I] #y_indexing
    y_I_squeeze = np.squeeze(y_I.toarray())
    sum_ = (X_I.dot(w) - y_I_squeeze).T @ (X_I.dot(w) - y_I_squeeze)
    return sum_



def get_loss(w, lambda_, X, y, n_batch):
    loss = 0
    batch_ = np.random.permutation(X.shape[0]) #shuffle
    for j in range(X.shape[0]//n_batch + 1):
        X_bc = X[batch_[j*n_batch:(j+1)*n_batch]] #make batch
        y_bc = y[batch_[j*n_batch:(j+1)*n_batch]] #make batch
        loss += slack_variable(w, X_bc, y_bc) #compute loss
    loss *= lambda_ / X.shape[0]
    loss += 1/2 * w.transpose().dot(w)
    return loss


def get_acc(w, lambda_,  X, y, n_batch):
    test_acc = 0
    batch_ = np.random.permutation(X.shape[0]) #shuffle
    for j in range(X.shape[0]//n_batch + 1):
        X_bc = X[batch_[j*n_batch:(j+1)*n_batch]] #make batch
        y_bc = y[batch_[j*n_batch:(j+1)*n_batch]].toarray() #make batch
        dcsn_bdry = X_bc.dot(w) > 0  #get decision boundary
        gd_trth = y_bc.reshape([-1]) > 0 #get ground truth
        test_acc += np.sum(dcsn_bdry == gd_trth)
    test_acc /= X.shape[0]
    return test_acc
