import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def proj_on_diag(x):
    return np.array([(x[0] + x[1]) / 2, (x[0] + x[1]) / 2])


def random_dg(nb_points):
    A = np.zeros((nb_points, 2))
    t = 0
    while t < nb_points:
        x = np.random.rand()
        y = np.random.rand()
        if y > x:
            A[t] = np.array([x, y])
            t += 1
    return A


def plot_axes(ax):
    ax.plot([0, 1], [0, 1])


def cost_matrix(d_1, d_2, norm):
    """
    :param d_1: first diagram, pdiag object
    :param d_2: second diagram, pdiag object
    :param norm: ground norm, basically L2 or L_infty
    :return: The cost matrix with size (k x k) where k = |d_1| + |d_2| in order to encode matching to diagonal

    Remark : this is a Lagrangian cost matrix, not the one used in Sinkhorn with Eulerian setting.
    """
    n, m = len(d_1.points), len(d_2.points)
    k = n + m
    M = np.zeros((k, k))
    for i in range(n):  # go throught d_1 points
        x_i = d_1.points[i]
        p_x_i = proj_on_diag(x_i)  # proj of x_i on the diagonal
        dist_x_delta = norm(x_i, p_x_i)  # distance to the diagonal regarding the ground norm
        for j in range(m):  # go throught d_2 points
            y_j = d_2.points[j]
            p_y_j = (proj_on_diag(y_j))
            M[i, j] = norm(x_i, y_j)
            dist_y_delta = norm(y_j, p_y_j)
            for it in range(m):
                M[n + it, j] = dist_y_delta
        for it in range(n):
            M[i, m + it] = dist_x_delta

    return M


class Pdiag:
    def __init__(self, filename=None, points=None, nb_points=10):
        if filename is not None:  # Priority to load the file
            try:
                a = np.atleast_2d(np.loadtxt(filename))
            except:
                a = np.atleast_2d(np.load(filename))
            self.points = a
        elif points is not None:
            self.points = points
        else:
            self.points = random_dg(nb_points)

    def save(self, filename, format='npy'):
        if format == 'npy':
            np.save(filename, self.points)
        elif format=='txt':
            np.savetxt(filename, self.points)
        else:
            raise Exception("Format unknown. Must be npy or txt.")

    def copy(self):
        x = self.points
        return Pdiag(points=x)


    def plot_dg(self, ax=None, marker='x', color='blue', block=False):
        """
        :param ax: an axex matplotlib object.
        :param marker: The kind of marker we want ('x' for cross, 'o' for dot, etc.)
        :param color: color of points
        :param block: use False if you want to be able to plot other diagram in same plot.
        :return: Plot a diagram in a fancy way.

        TODO : improve xlim.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        plot_axes(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.scatter(self.points[:, 0], self.points[:, 1], c=color, marker=marker, s=50)
        ax.add_patch(Polygon([[0,0], [1,0], [1,1]], fill=True, color='grey'))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        plt.show(block=block)
