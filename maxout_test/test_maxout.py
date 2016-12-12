import h5py
import numpy as np
import activation as act
import networkhelper as N

def t_maxout():
    network = N.Network()

    network.setInput(N.RawInput((1, 28,28)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=128))
    network.append(N.FeaturePooling(4))
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=8))
    network.append(N.FeaturePooling(4))
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=8))
    network.append(N.FeaturePooling(4))
    network.append(N.GlobalPooling())
    network.append(N.FullConn(input_feature=128, output_feature=10))
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

if __name__=="__main__":
    maxout = t_maxout()
    np.save('maxout',maxout)
