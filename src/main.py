import matplotlib.pylab as plt
import numpy as np
import os
from mnist import load_mnist  # Load MNIST
from PIL import Image

# Data
img_train = []   # 60000 Images, Each Has Length of 28*28=728
label_train = [] # 60000 Integers, Each from range(0, 19)
img_test = []    # 10000 Images
label_test = []  # 10000 Integers

# Function - Init
def init():

    # Global Variables
    global img_train, label_train, img_test, label_test

    # Image & Label
    (img_train, label_train), (img_test, label_test) = load_mnist(
        flatten=True, normalize=False)

# Function - Numerical Differentiation
def num_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# Function - Numerical Gradient
def num_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size):
        tmp_val = x[i]
        x[i] = tmp_val + h
        fxh1 = f(x)
        x[i] = tmp_val - h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp_val
    return grad

# Function - Gradient Descent
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = num_gradient(f, x)
        x -= lr * grad
    return x

# Main
if __name__ == "__main__":

    # Init
    init()