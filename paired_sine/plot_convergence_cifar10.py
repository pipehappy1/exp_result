import numpy as np
from matplotlib import pyplot as plt

def mnist_shallow():
    mnist_shallow = np.load('cifar10_shallow.npy')
    mnist_shallow_sin = np.load('cifar10_shallow_sin.npy')
    mnist_shallow_cos = np.load('cifar10_shallow_cos.npy')
    
    plt.plot(range(100), mnist_shallow[:100],     linestyle='-.', label='ReLU')
    plt.plot(range(100), mnist_shallow_sin[:100], label='sin')
    plt.plot(range(100), mnist_shallow_cos[:100], linestyle='--', label='cos')
    
    plt.legend(loc='best', frameon=False)
    
    plt.xlabel("epoch #")
    plt.ylabel("error rate")
    plt.title("Error rate vs epoch for shallow network on CIFAR10")
    
    plt.savefig('cifar10_convergence_shallow.pdf')

def mnist_deep():
    mnist_deep = np.load('mnist_deep.npy')
    mnist_deep_sin = np.load('mnist_deep_sin.npy')
    mnist_deep_cos = np.load('mnist_deep_cos.npy')
    
    plt.plot(range(100), mnist_deep[:100],     linestyle='-.', label='ReLU')
    plt.plot(range(100), mnist_deep_sin[:100], label='sin')
    plt.plot(range(100), mnist_deep_cos[:100], linestyle='--', label='cos')
    
    plt.legend(loc='best', frameon=False)
    
    plt.xlabel("epoch #")
    plt.ylabel("error rate")
    plt.title("Error rate vs epoch for deep network on MNIST")
    
    plt.savefig('mnist_convergence_deep.svg')


if __name__ == '__main__':
    mnist_shallow()
