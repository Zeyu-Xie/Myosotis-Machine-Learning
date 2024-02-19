import matplotlib.pylab as plt
import numpy as np
import os
from mnist import load_mnist # Load MNIST
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
    (img_train, label_train), (img_test, label_test) = load_mnist(flatten = True, normalize = False)

def f_1 (x):
    return 0.01*x**2+0.1*x

# Main
if __name__ == "__main__":

    # Init
    init()

    x = np.arange(0.0, 20.0, 0.1) # 以0.1为单位，从0到20的数组x
    y = f_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.savefig('output.png')