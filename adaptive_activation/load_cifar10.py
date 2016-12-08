import os
import numpy as np
import pickle
import theano
import theano.tensor as T

def load_cifar10(batch):
    fp = '/home/xtop/workspace/cnn/adaptive_activation/CIFAR10/cifar-10-batches-py/'
    fp = os.path.join(fp, str(batch))

    f = open(fp, 'rb')
    d = pickle.load(f, encoding="latin")
    f.close()

    data = d["data"]*0.01
    labels = d['labels']
    
    data = data.reshape(data.shape[0], 3*32*32)

    teY = np.zeros((len(labels), 10))
    teY[np.arange(len(labels)), labels] = 1
    
        
    #Transforms the label matrix into one-dimensional
    def lable_to_vector(y):
        vy = ([])
        for row in range(y.shape[0]):
            index = 0
            for column in range(y.shape[1]):
                index += 1
                if y[row][column] == 1:
                    vy.append(index-1)            
        return vy

    shared_x, shared_y = data,lable_to_vector(teY)
    return shared_x, shared_y

#splice two lable
def splice_lable(y1,y2):
    for x in y2:
        y1.append(x)
    return y1

#images1,lables1 = load_cifar10('data_batch_1')
#images2,lables2 = load_cifar10('data_batch_2')
#images3,lables3 = load_cifar10('data_batch_3')
#images4,lables4 = load_cifar10('data_batch_4')
#images5,lables5 = load_cifar10('data_batch_5')
#images = np.vstack((images1,images2,images3,images4,images5))
#lables = splice_lable(lables1,lables2)
#lables = splice_lable(lables,lables3)
#lables = splice_lable(lables,lables4)
#lables = splice_lable(lables,lables5)
#    
#train_set_x,train_set_y = images,lables
#valid_set_x,valid_set_y = images,lables
#test_set_x,test_set_y = load_cifar10('test_batch')