
import numpy as np
from numpy import random
import numbers
from math import pi
import matplotlib.pyplot as plt

from numpy import sin, cos
from scipy.special import gamma, multigammaln, gammaln
from scipy.stats import wishart
from numpy.random import multivariate_normal
from scipy.stats import multivariate_normal as norm
from numpy.linalg import inv, det, cholesky
import scipy
import matplotlib as mpl
import imageio, os

import torch

def dirichlet_expectation_k_torch(alpha, k):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return torch.subtract(torch.digamma(torch.add(alpha[k], np.finfo(np.float64).eps)),torch.digamma(torch.squeeze(alpha)))

def dirichlet_expectation_torch(alpha):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return torch.subtract(torch.digamma(torch.add(alpha, np.finfo(np.float64).eps)),torch.digamma(torch.squeeze(alpha)))



def log_beta_function_torch(x):
    """
    Log beta function
    ln(\gamma(x)) - ln(\gamma(\sum_{i=1}^{N}(x_{i}))
    """
    return torch.subtract(
        torch.squeeze(torch.lgamma(torch.add(x, np.finfo(np.float64).eps))),
        torch.lgamma(torch.squeeze(torch.add(x, np.finfo(np.float64).eps))))





def softmax_torch(x, axis=1):
    """
    Softmax computation
    e^{x} / sum_{i=1}^{K}(e^x_{i})
    """
    if len(x.shape)==1:
        x=torch.unsqueeze(x, axis=0)
    return torch.divide(torch.add(torch.exp(torch.subtract(x, torch.max(x, axis=axis, keepdims=True))),np.finfo(np.float64).eps),
    torch.sum(torch.add(torch.exp(torch.subtract(x, torch.max(x, axis=axis, keepdims=True))),np.finfo(np.float64).eps), axis=axis, keepdims=True))




def generate_random_positive_matrix(D):
    """
    Generate a random semidefinite positive matrix
    :param D: Dimension
    :return: DxD matrix
    """
    aux = random.rand(D, D)
    return np.dot(aux, aux.transpose())


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def draw_ellipse(mu, cov, num_sd=1):
    # as I understand it -
    # diagonalise the cov matrix
    # use eigenvalues for radii
    # use eigenvector rotation from x axis for rotation
    x = mu[0]  # x-position of the center
    y = mu[1]  # y-position of the center
    cov = cov * num_sd  # show within num_sd standard deviations
    lam, V = np.linalg.eig(cov)  # eigenvalues and vectors
    t_rot = np.arctan(V[1, 0] / V[0, 0])
    a, b = lam[0], lam[1]
    t = np.linspace(0, 2 * np.pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    # u,v removed to keep the same center location
    R_rot = np.array([[cos(t_rot), -sin(t_rot)], [sin(t_rot), cos(t_rot)]])
    # 2-D rotation matrix

    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
    return x + Ell_rot[0, :], y + Ell_rot[1, :]

cols = [\
        '#8159a4',
        '#60c4bf',
        '#f19c39',
        '#cb5763',
        '#6e8dd7',
        ]

def plot_GMM(X, mu, lam, pi, centres, covs, K, title, cols=cols, savefigpath=False):
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'kx', alpha=0.2)

    legend = ['Datapoints']

    for k in range(K):
        cov = lam[k]
        x_ell, y_ell = draw_ellipse(mu[k], cov)
        plt.plot(x_ell, y_ell, cols[k])
        legend.append('pi=%.2f, var1=%.2f, var2=%.2f, cov=%.2f' % (pi[k], cov[0, 0], cov[1, 1], cov[1, 0]))
        # for whatever reason lots of the ellipses are very long and narrow, why?

    # Plotting the ellipses for the GMM that generated the data
    for i in range(centres.shape[0]):
        x_true_ell, y_true_ell = draw_ellipse(centres[i], covs[i])
        plt.plot(x_true_ell, y_true_ell, 'g--')
        legend.append(
            'Data generation GMM %d, var1=%.2f, var2=%.2f, cov=%.2f' % (i, covs[i][0, 0], covs[i][1, 1], covs[i][1, 0]))

    plt.title(title)
    plt.plot(mu[:,0], mu[:,1], 'ro')
    if isinstance(savefigpath, str):
        plt.savefig(savefigpath)
    else:
        plt.legend(legend)
        plt.show()




def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
