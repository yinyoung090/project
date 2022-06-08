#in[1]
#Python >=3.5 is required
import sys
assert sys.version_info >= (3, 5)

#Scikit-Learn >=0.20 is required
import sklearn
assert sklearn.__version__ >= "0.2"

try:
    #%tensorflow_version only esists in Colab.
    %tensorflow_versoin 2.X
except Exception:
    pass

#Tensorflow >=2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

%load_ext tensorboard

#Common imports
import numpy as np
import os
#to make this notebook's output stable across runs
np.random.seed(42)

#To plot pretty figures
%matplotlib inline 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

#Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deep"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saveing figure" , fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format = fig_extension, dpi=resolution)

#Vanishing/Exploding Gradients Problem
#in[2]
def logit(z):
    return 1/ (1 + np.exp(-z))

#in[3]
z = np.linspace(-5, 5, 200)

plt.plot([-5, 5], [0, 0], "k-")
plt.plot([-5, 5], [1, 1], 'k--')
plt.plot([0, 0], [ -0.2, 1.2], 'k-')
plt.plot([-5, 5], [-3/4, 7/4], 'g--')
plt.plot(z, logit(z), "b-", linewidth=2)
props = dict(facecolor='black', shrink=0.1)
plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha='center')
plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
plt.annotate("Linear", xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center" )
plt.grid(True)
plt.title("Sigmond activation function", fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])

save_fig("sigmond_saturation_plot")
plt.show()

#Xavier and He Initialization
#in[4]
[name for name in dir(keras.initializers) if not name.startswith("_")]

#in[5]
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")

#in[6]
init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg',
                                        distribution='uniform')
keras.layers.Dense(10, activation="relu", kernel_initializer=init)

#Nonsaturating Activation Functions
#Leaky ReLU
#in[7]
def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha *z, z)

#in[8]
plt.plot(z, leaky_relu(z, 0.05), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([0, 0], [-0.5, 4.2], 'k-')
plt.grid(True)
props = dict(facecolor='black', shrink=0.1)
plt.annotate('Leak', xytext=(-4, 0.5), xy=(-0.5, -0.2), arrowprops=props, fontsize=14, ha="center")
plt.title("Leaky ReLU activation function", fontsize=14)
plt.axis([-5, 5, -0.5, 4.2])

save_fig("leaky_relu_plot")
plt.show()

#in[9]
[m for m in dir(keras.activations) if not m.startswith("_")]

#in[10]
[m for m in dir(keras.layers) if "relu" in m.lower()]

#in[11]
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#in[12]
tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(10, activation="softmax")
])

#in[13]
model.compile(loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.SGD(learning_rate=1e-3),
            metrics=["accuracy"])

#in[14]
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))

#Now let's try PReLU
#in[15]
tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential([ 
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])

#in[16]
model.compile(loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    metrics=["accuracy"]
)

#in[17]
history = model.fit(X_train, y_train, epochs=10,
                validation_data=(X_valid, y_valid))

#ELU
#in[18]
def elu(z, alpha=1):
    return np.where(z<0, alpha * (np.exp(z)-1), z)

#in[19]
plt.plot(z, elu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1, -1], 'k--')
plt.plot([0, 0], [-2.2, -2.2], 'k-')
plt.grid(True)
plt.title(r"ELU activation function ($\alpha=1$)", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])

#in[20]
keras.layers.Dense(10, activation="elu")

#SELU
#[21]
from scipy.special import erfc

#alpha and scale to self normalize with mean 0 and standard deciaiton 1
#(see equation 14 in the pater):
alpha_0_1 = -np.sqrt(2 / np.pi) / (erfc(1/np.sqrt(2)) * np.exp(1/2) - 1)
scale_0_1 = (1 - erfc(1 / np.sqrt(2)) * np.sqrt(np.e)) * np.sqrt(2 * np.pi) * (2 * erfc(np.sqrt(2))*np.e**2 + np.pi*erfc(1/np.sqrt(2))**2*np.e - 2*(2+np.pi)*erfc(1/np.sqrt(2))*np.sqrt(np.e)+np.pi+2)**(-1/2)

#in[22]
def selu(z, scale=scale_0_1, alpha=alpha_0_1):
    return scale * elu(z, alpha)

#in[23]
plt.plot(z, selu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1.758, -1.758], 'k--')
plt.grid(True)
plt.title("SELU activation function", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])

save_fig("selu_plot")
plt.show()

#in[24]
np.random.seed(42)
Z = np.random.normal(size=(500, 100)) # standardized inputs
for layer in range(1000):
    W = np.random.normal(size=(100, 100), scale=np.sqrt(1/100)) # LeCun initializaiton)
    Z = selu(np.dot(Z, W))
    means = np.mean(Z, axis=0).mean()
    stds = np.std(Z, axis=0).mean()
    if layer % 100 == 0:
        print("Layer {}: mean{:.2f}, std deciation {:.2f}".format(layer, means, stds))
        print("Z.shape is {}".format(Z.shape))

#in[25]
keras.layers.Dense(10, activation="selu",
                    kernel_initializer="lecun_normal")

#in[26]
np.random.seed(42)
tf.random.set_seed(42)

#in[27]
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="selu",
                                kernel_initializer="lecun_normal"))
for layer in range(99):
    model.add(keras.layers.Dense(100, activation="relu",
                                kernel_initializer="lecun_normal"))
model.add(keras.layers.Dense(10, activation="softmax"))

#in[28]
model.compile(loss="sparse_categorical_crossentropy",
                optimizer = keras.optimizers.SGD(learning_rate=1e-3),
                metrics=["accuracy"])

#in[29]
pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

#in[30]
history = model.fit(X_train_scaled, y_train, epochs=5, 
                    validation_data=(X_valid_scaled, y_valid))

#look at what happen if we try to use ReLU
#in[31]
np.random.seed(42)
tf.random.set_seed(42)

#in[32]
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu", kernel_initializer='he_normal'))
for layer in range(99):
    model.add(keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(10, activation="softmax"))

#in[33]
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                metrics=["accuracy"])

#in[34]
history = model.fit(X_train_scaled, y_train, epochs=5,
                validation_data=(X_valid_scaled, y_valid))

#Batch normalization
#in[35]
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.
])