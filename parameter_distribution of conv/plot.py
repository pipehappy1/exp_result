import numpy as np  
import matplotlib.pyplot as plt  

def load_array(address, low, stepsize):
    a = np.load(address)
    print(a.shape)
    a = a.reshape(a.shape[0]*a.shape[1]*a.shape[2]*a.shape[3])
    num = 0
    l = ([])
    for i in range(10):
        v = (np.sum(a<(low+stepsize*(i+1)))-np.sum(a<(low+stepsize*i)))
        #print(v)
        l.append(v)
        num+=v
    print(num)
    amax = a.max()
    amin = a.min()
    return l, a, amax, amin, low, stepsize

def draw_bar(labels,quants,amax,amin,low,stepsize):  
    width = 0.4  
    ind = np.linspace(0.5,9.5,10)  
    # make a square figure  
    fig = plt.figure(1)  
    ax  = fig.add_subplot(111)  
    # Bar Plot  
    ax.bar(ind-width/2,quants,width,color='green',align="center")  
    # Set the ticks on x-axis  
    ax.set_xticks(ind)  
    ax.set_xticklabels(labels)  
    # labels  
    ax.set_xlabel('range/'+"%s" % stepsize)  
    ax.set_ylabel('number')  
    # title  
    ax.set_title('w_distribution of conv9 on deep_cifar10', bbox={'facecolor':'0.9', 'pad':2})
    plt.text(7,np.asarray(quants).max()*9/10,'max:'+"%s" % amax)
    plt.text(7,np.asarray(quants).max()*4/5,'min:'+"%s" % amin)
    for i in range(10):
#        if i == 4:
#            plt.text(0.5+i, quants[i]-10000, "%s" % quants[i])
#            continue
        plt.text(0.1+i, (np.asarray(quants).max())/100.0+quants[i], "%s" % quants[i])
    plt.grid(True)
    plt.show()
  
quants, a, amax, amin, low, stepsize = load_array('/home/xtop/workspace/team_work/---/w of conv/deep_cifar10/deep_cifar10_conv9_w.npy', -0.08, 0.017)
labels   = ['%s~%s'% (low,round((low+stepsize),3)), '', '', '', '', '', '', '', '', '%s~%s'% (round((low+9*stepsize),3),round((low+10*stepsize),3))]  
draw_bar(labels,quants,amax,amin,low,stepsize)
