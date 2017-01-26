import mlbase.networkhelper as N
import h5py
import numpy as np
import mlbase.loaddata as l
import mlbase.activation as act
import mlbase.scheduling as sc
import sys

def test2():
    network = N.Network()
    network.modelPrefix = 'cifar10_shallow'
    network.learningRate = 0.001
    network.saveInterval = 0

    network.setInput(N.RawInput((3, 32,32)))
    network.append(N.Conv2d(feature_map_multiplier=32))
    network.append(act.Relu())
    network.append(N.Pooling())
    network.append(N.Conv2d(feature_map_multiplier=2))
    network.append(act.Relu())
    network.append(N.Pooling())
    network.append(N.Conv2d(feature_map_multiplier=2))
    network.append(act.Relu())
    network.append(N.Pooling())
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=6144, output_feature=6144*2))
    network.append(act.Relu())
    network.append(N.FullConn(input_feature=6144*2, output_feature=10))
    network.append(N.SoftMax())

    network.build()

    trX, trY, teX, teY = l.load_cifar10()

    savedErrorRate = []
    for i in range(500):
        network.train(trX, trY)
        errorRate = 1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1))
        savedErrorRate.append(errorRate)
        print(errorRate)
        sys.stdout.flush()

    np.save(network.modelPrefix, savedErrorRate)

if __name__ == '__main__':
    job = sc.Job(test2)
    job.run()