import numpy as np
import scipy.spatial.distance as sc

try:
    import ot
except ImportError:
    print("POT not found---DiagramQuantization not available")


def dist_to_diag(x, p=2.):
    """
    :param x: Point in R^2 verifying x[1] > x[0]
    :param p: exponent for dist to diag (default = 2)
    :return: d its distance to the diagonal to the power p
    """
    return 1/np.power(2, p/2) * np.power(abs(x[1] - x[0]), p)


def greed_init(Y,n,k):
    X = Y[np.random.choice(n, k)]  # randomly initialize positions
    d = np.mean(sc.pdist(X))
    for _ in range(100):
        X2 = Y[np.random.choice(n,k)]
        d2 = np.mean(sc.pdist(X2))
        if d2 > d:
            d = d2
            X = X2.copy()
    return X


def balanced_kmeans(Y, b, k, t, gamma, nb_max_iter, stopping_criterion, verbose):
    '''
    :param Y: n x 2 np.array ; encoding support of the input (weighted) measure (dgm)
    :param b: n np.array ; weights on each point (eg distance to diag)
    :param k: int ; number of centroid
    :param t: float in (0,1) ; similar to a learning rate
    :param gamma: float (non-negative) ; parameter to use in Sinkhorn approximation of opt transport plan. If 0, exact transport is computed.
    :param nb_max_iter: int ; nombre max iter in the process
    :param stopping_criterion: float ; stopping criterion of the process
    :return: X, k x 2 np.array ; encoding balanced kmeans.
    :return: P, k x n np.array ; optimal transport plan from X to Y.

    Inspired from [Fast Wasserstein barycenters ; Cuturi, Doucet, 2014]. Only true in euclidean.

    TODO : add line search method to set t
    TODO : improve stopping criterion (use W_2 metric)
    '''
    n = Y.shape[0]
    assert (Y.shape[1] == 2)
    assert (b.shape[0] == n)
    b = b/np.sum(b)  # Normalization of input mesure (to 1). This is necessary to use POT, and has *no* impact on quantization
                     #    since computed cells are mass-scale invariant.
    # X = Y[np.random.choice(n, k)]  # randomly initialize kmeans
    X = greed_init(Y, n, k)
    a = 1/k * np.ones(k)  # Uniform weights enforced

    for i in range(nb_max_iter):
        C = sc.cdist(X,Y)**2
        if gamma > 0:  # We apply sinkhorn reg
            P = ot.bregman.sinkhorn(a, b, C, gamma)  # size k x n
        else:  # exact computation of OT (ok for n not too large)
            P = ot.emd(a, b, C)
        new_X = (1 - t) * X + t * k * np.dot(P,Y)
        e = np.linalg.norm(new_X - X)
        if e < stopping_criterion:
            if verbose:
                print("nb iter done:", i)
            break
        else:
            X = new_X
    ### Mathieu: small modif, for now just return the diagram
    return X #, P


def weight_optim(hat_a, hat_b, C, t, gamma, nb_max_iter, stopping_criterion, verbose):
    '''
    :param a: k+1 np.array ; weights on each point of the (current estimate of) quantization
    :param hat_b: n+1 np.array ; weights on each point (eg distance to diag)
    :param C: the (k+1, n+1) cost matrix (previously computed, include the diagonal as a trash bin)
    :param t: float similar to a learning rate
    :param gamma: float (non-negative) ; parameter to use in Sinkhorn approximation of opt transport plan. If 0, exact transport is computed.
    :param nb_max_iter: int ; nombre max iter in the process
    :param stopping_criterion: float ; stopping criterion of the process
    :return: new_a, k np.array ; encoding new weights for the current estimate.

    Inspired from [Fast Wasserstein barycenters ; Cuturi, Doucet, 2014].
    	Adapted to pers dgm
    '''

    k = len(hat_a) - 1

    for i in range(nb_max_iter):
        if gamma > 0:
            dico = ot.bregman.sinkhorn(hat_a, hat_b, C, gamma, log=True)[1]
        else:
            dico = ot.emd(hat_a, hat_b, C, log=True)[1]
        
        alpha = dico['u'][:k]
        a = hat_a[:k]
        new_a = np.multiply(a, np.exp(- t * alpha))
        new_a = new_a/(2 * np.sum(new_a))

        e = np.mean(np.abs(new_a - a))

        if e < stopping_criterion:
            break
        else:
            hat_a = np.append(new_a, 0.5)

    return hat_a

def build_dist_matrix(X,Y):
    '''
    TODO : adapt to p (not only 2)
    '''
    C = sc.cdist(X,Y)**2
    Cxd = (X[:,1] - X[:,0])**2 / 2
    Cf = np.hstack((C, Cxd[:,None]))
    Cdy = (Y[:,1] - Y[:,0])**2 / 2
    Cdy = np.append(Cdy, 0)
    Cf = np.vstack((Cf, Cdy[None,:]))
    return Cf

def loc_update(hat_a, hat_b, Y, C, gamma):
    '''
    :param a: np.array ; size k, weights on X locations
    :param Y: np.array ; size n x 2 , location of point in attach data
    :param P: np.array ; size (k+1) x (n+1) , optimal (partial) transport plan between X and Y (and the diag)
    :return: updated position
    '''
    k = len(hat_a) - 1
    a = hat_a[:k]

    if gamma > 0:  # We apply sinkhorn reg
        P = ot.bregman.sinkhorn(hat_a, hat_b, C, gamma)  # size k x n
    else:  # exact computation of OT (ok for n not too large)
        P = ot.emd(hat_a, hat_b, C)

    k= P.shape[0] -1
    n = P.shape[1] -1
    Pxy = P[:k,:n]  # size k x n
    new_X = np.divide(Pxy.dot(Y), a[:,None])  # size k

    Y_mean = (Y[:,0] + Y[:,1]) / 2  # size n

    t = 1/np.sum(Pxy, axis=1) - 1/a  # size k
    new_X = new_X + np.multiply(t, Pxy.dot(Y_mean))[:,None]

    return new_X, P
