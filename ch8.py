#in[1]
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

#Projection methods
#Build 3D dataset
#in[2]
m = 60  
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.randn(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:,0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m)/2
X[:,1] = np.sin(angles)*0.7 + noise * np.random.randn(m)/2
X[:,2] = X[:,0]*w1 + X[:, 1]*w2 + noise *np.random.randn(m)

#PCA using SVD decomposition
#in[3]
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

#in[4]
m, n = X.shape

S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)

#in[5]
np.allclose(X_centered, U.dot(S).dot(Vt))

#in[6]
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)

#in[7]
X2D_using_svd = X2D


#PCA using Scikit-Learn
#in[8]
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)