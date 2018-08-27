"""
@author: Mathieu Carriere
All rights reserved
"""

import sys
import numpy as np
import scipy.spatial.distance as sc

from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    import ot

except ImportError:
    print("POT not found---DiagramQuantization not available")

try:
    from .vectors import *
    from .kernels import *
    from .hera_wasserstein import *
    from .hera_bottleneck import *
    USE_CYTHON = True

except ImportError:
    USE_CYTHON = False
    print("Cython not found---WassersteinDistance not available")

#############################################
# Preprocessing #############################
#############################################

# Method 2
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

class BirthPersistenceTransform(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return np.tensordot(X, np.array([[1.0, -1.0],[0.0, 1.0]]), 1)


class DiagramPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, use = False, scaler = StandardScaler()):
        self.scaler = scaler
        self.use    = use

    def fit(self, X, y = None):
        if self.use == True:
            if len(X) == 1:
                P = X[0]
            else:
                P = np.concatenate(X,0)
            self.scaler.fit(P)
        return self

    def transform(self, X):
        if self.use == True:
            Xfit, num_diag = [], len(X)
            for i in range(num_diag):
                diag = X[i]
                if diag.shape[0] > 0:
                    diag = self.scaler.transform(diag)
                Xfit.append(diag)
        else:
            Xfit = X
        return Xfit

class ProminentPoints(BaseEstimator, TransformerMixin):

    def __init__(self, use = False, num_pts = 10, threshold = -1):
        self.num_pts    = num_pts
        self.threshold  = threshold
        self.use        = use

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        if self.use == True:
            Xfit, num_diag = [], len(X)
            for i in range(num_diag):
                diag = X[i]
                if diag.shape[0] > 0:
                    pers       = np.matmul(diag, [-1.0, 1.0])
                    idx_thresh = pers >= self.threshold
                    thresh_diag, thresh_pers  = diag[idx_thresh.flatten()], pers[idx_thresh.flatten()]
                    sort_index  = np.flip(np.argsort(thresh_pers, axis = None),0)
                    sorted_diag = thresh_diag[sort_index[:min(self.num_pts, diag.shape[0])],:]
                    Xfit.append(sorted_diag)
                else:
                    Xfit.append(diag)
        else:
            Xfit = X
        return Xfit

class DiagramSelector(BaseEstimator, TransformerMixin):

    def __init__(self, limit = np.inf, point_type = "finite"):
        self.limit, self.point_type = limit, point_type

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        Xfit, num_diag = [], len(X)
        if self.point_type == "finite":
            for i in range(num_diag):
                diag = X[i]
                if diag.shape[0] != 0:
                    idx_fin = diag[:,1] != self.limit
                    Xfit.append(diag[idx_fin,:])
                else:
                    Xfit.append(diag)
        if self.point_type == "essential":
            for i in range(num_diag):
                diag = X[i]
                if diag.shape[0] != 0:
                    idx_ess = diag[:,1] == self.limit
                    Xfit.append(np.reshape(diag[:,0][idx_ess],[-1,1]))
                else:
                    Xfit.append(diag[:,:1])
        return Xfit










#############################################
# Finite Vectorization methods ##############
#############################################

class PersistenceImage(BaseEstimator, TransformerMixin):

    def __init__(self, bandwidth = 1.0, weight = lambda x: 1,
                       resolution = [20,20], im_range = [np.nan, np.nan, np.nan, np.nan]):
        self.bandwidth, self.weight = bandwidth, weight
        self.resolution, self.im_range = resolution, im_range

    def fit(self, X, y = None):
        if np.isnan(self.im_range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.im_range = [mx, Mx, my, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_CYTHON == True:

                image = np.array(persistence_image(diagram, self.im_range[0], self.im_range[1], self.resolution[0], self.im_range[2], self.im_range[3], self.resolution[1], self.bandwidth, self.weight))

            else:

                w = np.ones(num_pts_in_diag)
                for j in range(num_pts_in_diag):
                    w[j] = self.weight(diagram[j,:])

                x_values, y_values = np.linspace(self.im_range[0], self.im_range[1], self.resolution[0]), np.linspace(self.im_range[2], self.im_range[3], self.resolution[1])
                Xs, Ys = np.tile((diagram[:,0][:,np.newaxis,np.newaxis]-x_values[np.newaxis,np.newaxis,:]),[1,self.resolution[1],1]), np.tile(diagram[:,1][:,np.newaxis,np.newaxis]-y_values[np.newaxis,:,np.newaxis],[1,1,self.resolution[0]])
                image = np.tensordot(w, np.exp((-np.square(Xs)-np.square(Ys))/(2*self.bandwidth*self.bandwidth))/(self.bandwidth*np.sqrt(2*np.pi)), 1)

            Xfit.append(image.flatten()[np.newaxis,:])

        return np.concatenate(Xfit,0)

class Landscape(BaseEstimator, TransformerMixin):

    def __init__(self, num_landscapes = 5, resolution = 100, ls_range = [np.nan, np.nan]):
        self.num_landscapes, self.resolution, self.ls_range = num_landscapes, resolution, ls_range

    def fit(self, X, y = None):
        if np.isnan(self.ls_range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.ls_range = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.ls_range[0], self.ls_range[1], self.resolution)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_CYTHON == True:

                Xfit.append(np.array(landscape(diagram, self.num_landscapes, self.ls_range[0], self.ls_range[1], self.resolution)).flatten()[np.newaxis,:])

            else:

                ls = np.zeros([self.num_landscapes, self.resolution])

                events = []
                for j in range(self.resolution):
                    events.append([])

                for j in range(num_pts_in_diag):
                    [px,py] = diagram[j,:]
                    min_idx = np.minimum(np.maximum(np.ceil((px          - self.ls_range[0]) / step_x).astype(int), 0), self.resolution)
                    mid_idx = np.minimum(np.maximum(np.ceil((0.5*(py+px) - self.ls_range[0]) / step_x).astype(int), 0), self.resolution)
                    max_idx = np.minimum(np.maximum(np.ceil((py          - self.ls_range[0]) / step_x).astype(int), 0), self.resolution)

                    if min_idx < self.resolution and max_idx > 0:

                        landscape_value = self.ls_range[0] + min_idx * step_x - px
                        for k in range(min_idx, mid_idx):
                            events[k].append(landscape_value)
                            landscape_value += step_x

                        landscape_value = py - self.ls_range[0] - mid_idx * step_x
                        for k in range(mid_idx, max_idx):
                            events[k].append(landscape_value)
                            landscape_value -= step_x

                for j in range(self.resolution):
                    events[j].sort(reverse = True)
                    for k in range( min(self.num_landscapes, len(events[j])) ):
                        ls[k,j] = events[j][k]

                Xfit.append(np.sqrt(2)*np.reshape(ls,[1,-1]))

        return np.concatenate(Xfit,0)

class Silhouette(BaseEstimator, TransformerMixin):

    def __init__(self, weight = lambda x: 1, resolution = 100, sh_range = [np.nan, np.nan]):
        self.weight, self.resolution, self.sh_range = weight, resolution, sh_range

    def fit(self, X, y = None):
        if np.isnan(self.sh_range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.sh_range = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.sh_range[0], self.sh_range[1], self.resolution)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_CYTHON == True:

                Xfit.append(np.array(silhouette(diagram, self.sh_range[0], self.sh_range[1], self.resolution, self.weight))[np.newaxis,:])

            else:

                sh, weights = np.zeros(self.resolution), np.zeros(num_pts_in_diag)
                for j in range(num_pts_in_diag):
                    weights[j] = self.weight(diagram[j,:])
                total_weight = np.sum(weights)

                for j in range(num_pts_in_diag):

                    [px,py] = diagram[j,:]
                    weight  = weights[j] / total_weight
                    min_idx = np.minimum(np.maximum(np.ceil((px          - self.sh_range[0]) / step_x).astype(int), 0), self.resolution)
                    mid_idx = np.minimum(np.maximum(np.ceil((0.5*(py+px) - self.sh_range[0]) / step_x).astype(int), 0), self.resolution)
                    max_idx = np.minimum(np.maximum(np.ceil((py          - self.sh_range[0]) / step_x).astype(int), 0), self.resolution)

                    if min_idx < self.resolution and max_idx > 0:

                        silhouette_value = self.sh_range[0] + min_idx * step_x - px
                        for k in range(min_idx, mid_idx):
                            sh[k] += weight * silhouette_value
                            silhouette_value += step_x

                        silhouette_value = py - self.sh_range[0] - mid_idx * step_x
                        for k in range(mid_idx, max_idx):
                            sh[k] += weight * silhouette_value
                            silhouette_value -= step_x

                Xfit.append(np.reshape(np.sqrt(2) * sh, [1,-1]))

        return np.concatenate(Xfit, 0)

class BettiCurve(BaseEstimator, TransformerMixin):

    def __init__(self, resolution = 100, bc_range = [np.nan, np.nan]):
        self.resolution, self.bc_range = resolution, bc_range

    def fit(self, X, y = None):
        if np.isnan(self.bc_range[0]) == True:
            pre = DiagramPreprocessor(use=True, scaler=MinMaxScaler()).fit(X,y)
            [mx,my],[Mx,My] = pre.scaler.data_min_, pre.scaler.data_max_
            self.bc_range = [mx, My]
        return self

    def transform(self, X):

        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.bc_range[0], self.bc_range[1], self.resolution)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            if USE_CYTHON == True:

                Xfit.append(np.array(betti_curve(diagram, self.bc_range[0], self.bc_range[1], self.resolution))[np.newaxis,:])

            else:

                bc =  np.zeros(self.resolution)
                for j in range(num_pts_in_diag):
                    [px,py] = diagram[j,:]
                    min_idx = np.minimum(np.maximum(np.ceil((px - self.bc_range[0]) / step_x).astype(int), 0), self.resolution)
                    max_idx = np.minimum(np.maximum(np.ceil((py - self.bc_range[0]) / step_x).astype(int), 0), self.resolution)
                    for k in range(min_idx, max_idx):
                        bc[k] += 1

                Xfit.append(np.reshape(bc,[1,-1]))

        return np.concatenate(Xfit, 0)

class TopologicalVector(BaseEstimator, TransformerMixin):

    def __init__(self, threshold = 10):
        self.threshold = threshold

    def fit(self, X, y = None):
        return self

    def transform(self, X):

        num_diag = len(X)
        Xfit = np.zeros([num_diag, self.threshold])

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]
            pers = 0.5 * np.matmul(diagram, np.array([[-1.0],[1.0]]))
            min_pers = np.minimum(pers,np.transpose(pers))
            distances = sc.pdist(diagram, metric="chebyshev")
            vect = np.flip(np.sort(np.triu(np.minimum(distances, min_pers)), axis = None), 0)
            dim = np.minimum(len(vect), self.threshold)
            Xfit[i, :dim] = vect[:dim]

        return Xfit











#############################################
# Kernel methods ############################
#############################################

def mergeSorted(a, b):
    l = []
    while a and b:
        if a[0] < b[0]:
            l.append(a.pop(0))
        else:
            l.append(b.pop(0))
    return l + a + b

class SlicedWasserstein(BaseEstimator, TransformerMixin):

    def __init__(self, num_directions = 10, bandwidth = 1.0):
        self.num_directions = num_directions
        self.bandwidth = bandwidth

    def fit(self, X, y = None):

        if USE_CYTHON == False:

            num_diag = len(X)
            angles = np.linspace(-np.pi/2, np.pi/2, self.num_directions + 1)
            self.step_angle_ = angles[1] - angles[0]
            self.thetas_ = np.concatenate([np.cos(angles[:-1])[np.newaxis,:], np.sin(angles[:-1])[np.newaxis,:]], 0)

            self.proj_, self.proj_delta_ = [], []
            for i in range(num_diag):

                diagram = X[i]
                diag_thetas, list_proj = np.tensordot(diagram, self.thetas_, 1), []
                for j in range(self.num_directions):
                    list_proj.append( list(np.sort(diag_thetas[:,j])) )
                self.proj_.append(list_proj)

                diagonal_diagram = np.tensordot(diagram, np.array([[0.5,0.5],[0.5,0.5]]), 1)
                diag_thetas, list_proj_delta = np.tensordot(diagonal_diagram, self.thetas_, 1), []
                for j in range(self.num_directions):
                    list_proj_delta.append( list(np.sort(diag_thetas[:,j])) )
                self.proj_delta_.append(list_proj_delta)

        self.diagrams_ = list(X)

        return self

    def transform(self, X):

        if USE_CYTHON == False:

            num_diag1 = len(self.proj_)

            if np.array_equal(np.concatenate(self.diagrams_,0), np.concatenate(X,0)) == True:

                Xfit = np.zeros( [num_diag1, num_diag1] )
                for i in range(num_diag1):
                    for j in range(i, num_diag1):

                        L1, L2 = [], []
                        for k in range(self.num_directions):
                            ljk, ljkd, lik, likd = list(self.proj_[j][k]), list(self.proj_delta_[j][k]), list(self.proj_[i][k]), list(self.proj_delta_[i][k])
                            L1.append( np.array(mergeSorted(ljk, likd))[:,np.newaxis] )
                            L2.append( np.array(mergeSorted(lik, ljkd))[:,np.newaxis] )
                        L1, L2 = np.concatenate(L1,1), np.concatenate(L2,1)

                        Xfit[i,j] = np.sum(self.step_angle_*np.sum(np.abs(L1-L2),0)/np.pi)
                        Xfit[j,i] = Xfit[i,j]

                Xfit =  np.exp(-Xfit/(2*self.bandwidth*self.bandwidth))

            else:

                num_diag2 = len(X)
                proj, proj_delta = [], []
                for i in range(num_diag2):

                    diagram = X[i]
                    diag_thetas, list_proj = np.tensordot(diagram, self.thetas_, 1), []
                    for j in range(self.num_directions):
                        list_proj.append( list(np.sort(diag_thetas[:,j])) )
                    proj.append(list_proj)

                    diagonal_diagram = np.tensordot(diagram, np.array([[0.5,0.5],[0.5,0.5]]), 1)
                    diag_thetas, list_proj_delta = np.tensordot(diagonal_diagram, self.thetas_, 1), []
                    for j in range(self.num_directions):
                        list_proj_delta.append( list(np.sort(diag_thetas[:,j])) )
                    proj_delta.append(list_proj_delta)

                Xfit = np.zeros( [num_diag2, num_diag1] )
                for i in range(num_diag2):
                    for j in range(num_diag1):

                        L1, L2 = [], []
                        for k in range(self.num_directions):
                            ljk, ljkd, lik, likd = list(self.proj_[j][k]), list(self.proj_delta_[j][k]), list(proj[i][k]), list(proj_delta[i][k])
                            L1.append( np.array(mergeSorted(ljk, likd))[:,np.newaxis] )
                            L2.append( np.array(mergeSorted(lik, ljkd))[:,np.newaxis] )
                        L1, L2 = np.concatenate(L1,1), np.concatenate(L2,1)

                        Xfit[i,j] = np.sum(self.step_angle_*np.sum(np.abs(L1-L2),0)/np.pi)

                Xfit =  np.exp(-Xfit/(2*self.bandwidth*self.bandwidth))

        else:

            Xfit = np.array(sliced_wasserstein_matrix(X, self.diagrams_, self.bandwidth, self.num_directions))

        return Xfit

class PersistenceWeightedGaussian(BaseEstimator, TransformerMixin):

    def __init__(self, bandwidth = 1.0, weight = lambda x: 1, use_pss = False):

        self.bandwidth = bandwidth
        self.weight    = weight
        self.use_pss   = use_pss

    def fit(self, X, y = None):

        self.diagrams_ = list(X)

        if self.use_pss == True:
            for i in range(len(self.diagrams_)):
                op_D = np.tensordot(self.diagrams_[i], np.array([[0.0,1.0],[1.0,0.0]]), 1)
                self.diagrams_[i] = np.concatenate([self.diagrams_[i], op_D], 0)

        if USE_CYTHON == False:
            self.w_ = []
            for i in range(len(self.diagrams_)):
                num_pts_in_diag = self.diagrams_[i].shape[0]
                w = np.ones(num_pts_in_diag)
                for j in range(num_pts_in_diag):
                    w[j] = self.weight(self.diagrams_[i][j,:])
                self.w_.append(w)

        return self

    def transform(self, X):

        Xp = list(X)
        if self.use_pss == True:
            for i in range(len(Xp)):
                op_X = np.tensordot(Xp[i], np.array([[0.0,1.0],[1.0,0.0]]), 1)
                Xp[i] = np.concatenate([Xp[i], op_X], 0)

        if USE_CYTHON == True:

            Xfit = np.array(persistence_weighted_gaussian_matrix(Xp, self.diagrams_, self.bandwidth, self.weight))

        else:

            num_diag1 = len(self.w_)

            if np.array_equal(np.concatenate(Xp,0), np.concatenate(self.diagrams_,0)) == True:

                Xfit = np.zeros([num_diag1, num_diag1])

                for i in range(num_diag1):
                    for j in range(i,num_diag1):

                        d1x, d1y, d2x, d2y = self.diagrams_[i][:,0][:,np.newaxis], self.diagrams_[i][:,1][:,np.newaxis], self.diagrams_[j][:,0][np.newaxis,:], self.diagrams_[j][:,1][np.newaxis,:]
                        Xfit[i,j] = np.tensordot(self.w_[j], np.tensordot(self.w_[i], np.exp( -(np.square(d1x-d2x) + np.square(d1y-d2y)) / (2*self.bandwidth*self.bandwidth)) / (self.bandwidth*np.sqrt(2*np.pi)), 1), 1)
                        Xfit[j,i] = Xfit[i,j]
            else:

                num_diag2 = len(Xp)
                w = []
                for i in range(num_diag2):
                    num_pts_in_diag = Xp[i].shape[0]
                    we = np.ones(num_pts_in_diag)
                    for j in range(num_pts_in_diag):
                        we[j] = self.weight(Xp[i][j,:])
                    w.append(we)

                Xfit = np.zeros([num_diag2, num_diag1])

                for i in range(num_diag2):
                    for j in range(num_diag1):

                        d1x, d1y, d2x, d2y = Xp[i][:,0][:,np.newaxis], Xp[i][:,1][:,np.newaxis], self.diagrams_[j][:,0][np.newaxis,:], self.diagrams_[j][:,1][np.newaxis,:]
                        Xfit[i,j] = np.tensordot(self.w_[j], np.tensordot(w[i], np.exp( -(np.square(d1x-d2x) + np.square(d1y-d2y)) / (2*self.bandwidth*self.bandwidth)) / (self.bandwidth*np.sqrt(2*np.pi)), 1), 1)

        return Xfit

class PersistenceScaleSpace(BaseEstimator, TransformerMixin):

    def __init__(self, bandwidth = 1.0):
        self.PWG = PersistenceWeightedGaussian(bandwidth = bandwidth, weight = lambda x: 1 if x[1] >= x[0] else -1, use_pss = True)

    def fit(self, X, y = None):
        self.PWG.fit(X,y)
        return self

    def transform(self, X):
        return self.PWG.transform(X)






#############################################
# Metrics ###################################
#############################################

def compute_wass_matrix(diags1, diags2, p = 1, delta = 0.001):

    num_diag1 = len(diags1)

    if np.array_equal(np.concatenate(diags1,0), np.concatenate(diags2,0)) == True:
        matrix = np.zeros((num_diag1, num_diag1))

        if USE_CYTHON == True:
            if np.isinf(p):
                for i in range(num_diag1):
                    sys.stdout.write( str(i*1.0 / num_diag1) + "\r")
                    for j in range(i+1, num_diag1):
                        matrix[i,j] = bottleneck(diags1[i], diags1[j], delta)
                        matrix[j,i] = matrix[i,j]
            else:
                for i in range(num_diag1):
                    sys.stdout.write( str(i*1.0 / num_diag1) + "\r")
                    for j in range(i+1, num_diag1):
                        matrix[i,j] = wasserstein(diags1[i], diags1[j], p, delta)
                        matrix[j,i] = matrix[i,j]
        else:
            print("Cython required---returning null matrix")

    else:
        num_diag2 = len(diags2)
        matrix = np.zeros((num_diag1, num_diag2))

        if USE_CYTHON == True:
            if np.isinf(p):
                for i in range(num_diag1):
                    sys.stdout.write( str(i*1.0 / num_diag1) + "\r")
                    for j in range(num_diag2):
                        matrix[i,j] = bottleneck(diags1[i], diags2[j], delta)
            else:
                for i in range(num_diag1):
                    sys.stdout.write( str(i*1.0 / num_diag1) + "\r")
                    for j in range(num_diag2):
                        matrix[i,j] = wasserstein(diags1[i], diags2[j], p, delta)
        else:
            print("Cython required---returning null matrix")

    return matrix

class WassersteinDistance(BaseEstimator, TransformerMixin):

    def __init__(self, wasserstein = 1, delta = 0.001):
        self.wasserstein = wasserstein
        self.delta = delta

    def fit(self, X, y = None):
        self.diagrams_ = X
        return self

    def transform(self, X):
        return compute_wass_matrix(X, self.diagrams_, self.wasserstein, self.delta)
