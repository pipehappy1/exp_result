import mlbase.networkhelper as N
import h5py
import numpy as np
import mlbase.loaddata as l
import mlbase.activation as act
import mlbase.scheduling as sc

def test2():
    network = N.Network()
    network.modelPrefix = 'mnist_shallow_sin'

    network.setInput(N.RawInput((1, 28,28)))
    network.append(N.Conv2d(feature_map_multiplier=32))
    network.append(act.Sine())
    network.append(N.Pooling())
    network.append(N.Conv2d(feature_map_multiplier=2))
    network.append(act.Sine())
    network.append(N.Pooling())
    network.append(N.Conv2d(feature_map_multiplier=2))
    network.append(act.Sine())
    network.append(N.Pooling())
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(act.Relu())
    network.append(N.FullConn(input_feature=1152*2, output_feature=10))
    network.append(N.SoftMax())

    network.build()

    trX, trY, teX, teY = l.load_mnist()

    savedErrorRate = []
    for i in range(500):
        network.train(trX, trY)
        errorRate = 1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1))
        savedErrorRate.append(errorRate)
        print(errorRate)

    np.save(network.modelPrefix, savedErrorRate)

if __name__ == '__main__':
    job = sc.Job(test2)
    job.run()