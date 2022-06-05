# python >=2.35 is required
import sys
assert sys.version_info >= (3, 5)

#scikit-learn >=0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

try:
    # %tensorflow_version only exist in Colab.
    %tensorflow_verison 2.x 
except Exception:
    pass

#tensorflow >= 2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

%load_est tensorboard

#common imports
import numpy as np
import os
#to make this notebook's output stable across runs
np.random.seed(42)

#to plot pretty figures
%matplotlib inline 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsiz=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#where to save the figures
PROJECT_ROOT_DIR = '.'
CHAPTER_ID = "deep"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'images", CHAPTER_ID')
os.makedirs(IMAGES_PATH, exist_os=True)

def save_fig(fig_id, tigh_layout=True, fig_extension='png', resolution=300):
    path=os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("saveing figure", fig_id)
    if tigh_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

#Vanishing/Exploding Gradients problem
#in[2]
def logit(z):
    return 1/(1 + np.exp(-z))

#in[3]
