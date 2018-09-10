import numpy as np
import scipy.spatial.distance as sc
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import ot
except ImportError:
    print("POT not found---DiagramQuantization not available")

from utils_quantization import dist_to_diag, weight_optim, build_dist_matrix, loc_update, balanced_kmeans, greed_init
import pdiag

#############################################
# Quantization ##############################
#############################################

### Theo's code ###

def balanced_quantize(Y, k,
                 weight_function = lambda x : dist_to_diag(x, 2.), t=0.5,
                 gamma=0., nb_max_iter=100, stopping_criterion=0.001, verbose=False):
    """
    :param Y: n x 2 np.array ; encoding support of the input dgm (not weighted yet)
    :param k: int ; number of centroid
    :param weight_function: lambda x : R^2 --> R_+ ; weight to apply to points in the dgm.
    :param t: float in (0,1) ; similar to a learning rate
    :param gamma: float (non-negative) ; parameter to use in Sinkhorn approximation of opt transport plan. If 0, exact transport is computed.
    :param nb_max_iter: int ; maximum iter in the process
    :param stopping_criterion: float ; stopping criterion of the process
    :return: X, k x 2 np.array ; encoding balanced kmeans.
    :return: P, k x n np.array ; optimal transport plan at the end of process.
    """
    b = np.array([weight_function(x) for x in Y])
    return balanced_kmeans(Y, b, k, t, gamma, nb_max_iter, stopping_criterion, verbose)


def kmeans_quantize(Y, k, weight_update=False, gamma=0., 
                    nb_max_iter=100, stopping_criterion=0.001, t=10, 
                     verbose=False):
    """
    :param Y: n x 2 np.array ; encoding support of the input dgm
    :param k: int ; number of centroid
    :param gamma: float (non-negative) ; parameter to use in Sinkhorn approximation of opt transport plan. If 0, exact transport is computed.
    :param nb_max_iter: int ; maximum iter in the process
    :param stopping_criterion: float ; stopping criterion of the process
    :return: X, k x 2 np.array ; encoding points positions
    :return: a, k x 1 np.array ; encoding points weights
    :return: P, k x n np.array ; optimal transport plan at the end of process.

    TODO : improve initialization
    TODO : improve weight update
    """
    n = Y.shape[0]
    assert (Y.shape[1] == 2)

    b = 1/(2 * n) * np.ones(n)  # weight vector of the input diagram. Uniform here.
    hat_b = np.append(b, 0.5)  # so that we have a probability measure

    X = greed_init(Y, n, k)

    a = 1/(2 * k) * np.ones(k)  # Uniform initialization of weight
    hat_a = np.append(a , 0.5)  # so that we have a probability measure

    for i in range(nb_max_iter):
        C = build_dist_matrix(X,Y)

        new_X, P = loc_update(hat_a, hat_b, Y, C, gamma)

        ### Compute error update
        diff = np.linalg.norm(new_X - X, axis=1)
        diff = diff[~np.isnan(diff)]
        e = np.max(diff)
        if e < stopping_criterion:
            break
        else:
            X = new_X

        if weight_update:
            hat_a = weight_optim(hat_a, hat_b, C, t, gamma, nb_max_iter, stopping_criterion, verbose)

    if verbose:
        print("nb iter done:", i+1)
   
    if weight_update:
        return new_X, P, 2*n * hat_a[:k]

    return new_X, P


### Mathieu's code ###

class DiagramQuantization(BaseEstimator, TransformerMixin):

    def __init__(self, centroids = 10, learning_rate = 0.1, weight = lambda x: x[1] - x[0], gamma = 0.0, max_iter = 100, stop = 0.01, method = 1):
        self.centroids, self.lr, self.weight, self.gamma, self.max_iter, self.stop = centroids, learning_rate, weight, gamma, max_iter, stop
        self.method = method

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        Xfit = []
        for i in range(len(X)):
            diagram = X[i]
            if self.method == 1:
                w = np.array([self.weight(p) for p in diagram])
                Xfit.append(balanced_quantize(Y=diagram, k=self.centroids, weight_function=self.weight, t=self.lr, gamma=self.gamma, nb_max_iter=self.max_iter, stopping_criterion=self.stop))
            if self.method == 2:
                Xfit.append(kmeans_quantize(Y=diagram, k=self.centroids, gamma=self.gamma, nb_max_iter=self.max_iter, stopping_criterion=self.stop))
        return Xfit
