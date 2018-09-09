import numpy as np
import scipy.spatial.distance as sc
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import ot
except ImportError:
    print("POT not found---DiagramQuantization not available")

#############################################
# Quantization ##############################
#############################################

# Method 2
## Coucou 
def build_dist_matrix(X,Y):
    C = sc.cdist(X,Y)
    Cxd = (X[:,1] - X[:,0])**2 / 2
    Cf = np.hstack((C, Cxd[:,None]))
    Cdy = (Y[:,1] - Y[:,0])**2 / 2
    Cdy = np.append(Cdy, 0)
    Cf = np.vstack((Cf, Cdy[None,:]))
    return Cf

def loc_update(a, Y, P):
    k= P.shape[0] -1
    n = P.shape[1] -1
    Pxy = P[:k,:n]
    new_X = np.divide(Pxy.dot(Y), a[:,None])

    Y_mean = (Y[:,0] + Y[:,1]) / 2
    t = 1.0/np.sum(Pxy, axis=1) - 1.0/a
    new_X = new_X + np.multiply(t, Pxy.dot(Y_mean))[:,None]

    return new_X

def kmeans_quantize(Y, k, gamma, nb_max_iter, stopping_criterion):
    n = Y.shape[0]
    b = 1.0/(2 * n) * np.ones(n)
    hat_b = np.append(b, 0.5)

    X = Y[np.random.choice(n, k)]
    a = 1.0/(2 * k) * np.ones(k)
    hat_a = np.append(a, 0.5)

    for i in range(nb_max_iter):

        C = build_dist_matrix(X,Y)

        if gamma > 0:
            P = ot.bregman.sinkhorn(hat_a, hat_b, C, gamma)
        else:
            P = ot.emd(hat_a, hat_b, C)

        new_X = loc_update(a, Y, P)

        e = np.mean(np.linalg.norm(new_X - X, axis=0))
        if e < stopping_criterion:
            break
        else:
            X = new_X

    return new_X


# Method 1
def balanced_quantize(Y, k, b, t, gamma, max_iter, stop):
    n = Y.shape[0]
    b = b*1.0/np.sum(b)
    X = Y[np.random.choice(n, k)]
    a = 1.0/k * np.ones(k)

    for i in range(max_iter):
        C = sc.cdist(X,Y)
        if gamma > 0:
            P = ot.bregman.sinkhorn(a, b, C, gamma)
        else:
            P = ot.emd(a, b, C)
        new_X = (1 - t) * X + t * k * np.dot(P,Y)
        e = np.mean(np.linalg.norm(new_X - X, axis=0))
        if e < stop:
            break
        else:
            X = new_X

    return new_X


class DiagramQuantization(BaseEstimator, TransformerMixin):

    def __init__(self, centroids = 10, learning_rate = 0.1, weight = lambda x: x[1] - x[0], gamma = 0.0, max_iter = 100, stop = 0.01):
        self.centroids, self.lr, self.weight, self.gamma, self.max_iter, self.stop = centroids, learning_rate, weight, gamma, max_iter, stop

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        Xfit = []
        for i in range(len(X)):
            diagram = X[i]
            w = np.array([self.weight(p) for p in diagram])
            Xfit.append(balanced_quantize(Y=diagram, b=w, k=self.centroids, t=self.lr, gamma=self.gamma, max_iter=self.max_iter, stop=self.stop))
            #Xfit.append(kmeans_quantize(Y=diagram, k=self.centroids, gamma=self.gamma, nb_max_iter=self.max_iter, stopping_criterion=self.stop))
        return Xfit
