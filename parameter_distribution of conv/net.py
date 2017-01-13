import networkhelper as N
import h5py
import pickle
import os
import numpy as np
import mlbase.activation as act

def shallow_mnist():
    network = N.Network()
    network.debug = True
    network.learningRate = 0.001
    network.setInput(N.RawInput((1, 28,28)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=32, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=1152, output_feature=1024, need_bias=True))
    network.append(act.Relu())
    network.append(N.FullConn(input_feature=1024, output_feature=10, need_bias=True))
    network.append(N.SoftMax())
    network.build()

    f = h5py.File('/hdd/home/largedata/MNIST/mnist.hdf5', 'r')

    trX = f['x_train'][:,:].reshape(-1, 1, 28, 28)
    teX = f['x_test'][:,:].reshape(-1, 1, 28, 28)

    trY = np.zeros((f['t_train'].shape[0], 10))
    trY[np.arange(len(f['t_train'])), f['t_train']] = 1
    teY = np.zeros((f['t_test'].shape[0], 10))
    teY[np.arange(len(f['t_test'])), f['t_test']] = 1
    
    list = ([])
    for i in range(100):
        print(i)
        network.train(trX, trY)
        error_rate = 1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1))
        list.append(error_rate)
        print(error_rate)
    return list

def deep_mnist():
    network = N.Network()
    network.debug = True
    network.learningRate = 0.001
    network.setInput(N.RawInput((1, 28,28)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=32, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=1152, output_feature=1024, need_bias=True))
    network.append(act.Relu())
    network.append(N.FullConn(input_feature=1024, output_feature=10, need_bias=True))
    network.append(N.SoftMax())
    network.build()

    f = h5py.File('/hdd/home/largedata/MNIST/mnist.hdf5', 'r')

    trX = f['x_train'][:,:].reshape(-1, 1, 28, 28)
    teX = f['x_test'][:,:].reshape(-1, 1, 28, 28)

    trY = np.zeros((f['t_train'].shape[0], 10))
    trY[np.arange(len(f['t_train'])), f['t_train']] = 1
    teY = np.zeros((f['t_test'].shape[0], 10))
    teY[np.arange(len(f['t_test'])), f['t_test']] = 1
    
    list = ([])
    for i in range(100):
        print(i)
        network.train(trX, trY)
        error_rate = 1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1))
        list.append(error_rate)
        print(error_rate) 
    return list

def shallow_cifar10():
    network = N.Network()
    network.debug = True
    network.learningRate = 0.0001
    network.setInput(N.RawInput((3, 32,32)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=32, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=6144, output_feature=1024, need_bias=True))
    network.append(act.Relu())
    network.append(N.FullConn(input_feature=1024, output_feature=10, need_bias=True))
    network.append(N.SoftMax())
    network.build()
    
    images1,lables1 = load_cifar10('data_batch_1')
    images2,lables2 = load_cifar10('data_batch_2')
    images3,lables3 = load_cifar10('data_batch_3')
    images4,lables4 = load_cifar10('data_batch_4')
    images5,lables5 = load_cifar10('data_batch_5')
    images = np.vstack((images1,images2,images3,images4,images5))
    lables = np.vstack((lables1,lables2,lables3,lables4,lables5))
    trX, trY = images,lables
    teX, teY = load_cifar10('test_batch')
    trX = trX.reshape(-1, 3, 32, 32)
    teX = teX.reshape(-1, 3, 32, 32)
    
    list = ([])
    for i in range(100):
        print(i)
        network.train(trX, trY)
        error_rate = 1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1))
        list.append(error_rate)
        print(error_rate) 
    return list

def deep_cifar10():
    network = N.Network()
    network.debug = True
    network.learningRate = 0.0001
    network.setInput(N.RawInput((3, 32,32)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=32, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=1, need_bias=True))
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=6144, output_feature=1024, need_bias=True))
    network.append(act.Relu())
    network.append(N.FullConn(input_feature=1024, output_feature=10, need_bias=True))
    network.append(N.SoftMax())
    network.build()
    
    images1,lables1 = load_cifar10('data_batch_1')
    images2,lables2 = load_cifar10('data_batch_2')
    images3,lables3 = load_cifar10('data_batch_3')
    images4,lables4 = load_cifar10('data_batch_4')
    images5,lables5 = load_cifar10('data_batch_5')
    images = np.vstack((images1,images2,images3,images4,images5))
    lables = np.vstack((lables1,lables2,lables3,lables4,lables5))
    trX, trY = images,lables
    teX, teY = load_cifar10('test_batch')
    trX = trX.reshape(-1, 3, 32, 32)
    teX = teX.reshape(-1, 3, 32, 32)
    
    list = ([])
    for i in range(100):
        print(i)
        network.train(trX, trY)
        error_rate = 1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1))
        list.append(error_rate)
        print(error_rate) 
    return list

def load_cifar10(batch):
    fp = '/hdd/home/largedata/CIFAR10/cifar-10-batches-py/'
    fp = os.path.join(fp, str(batch))

    f = open(fp, 'rb')
    d = pickle.load(f, encoding="latin")
    f.close()

    data = d["data"]
    labels = d['labels']
    
    data = data.reshape(data.shape[0], 3*32*32)

    teY = np.zeros((len(labels), 10))
    teY[np.arange(len(labels)), labels] = 1
    
    return data, teY

if __name__ == "__main__":
    shallow_mnist = shallow_mnist()
    #deep_mnist = deep_mnist()
    #shallow_cifar10 = shallow_cifar10()
    #deep_cifar10 = deep_cifar10()
