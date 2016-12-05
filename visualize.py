import matplotlib.pyplot as plt
import numpy as np
import configure as cfg
import os, ipdb
from sklearn.metrics import confusion_matrix


EXPERIMENT = 'depeval_99'
N_SAMPLES = 100

if __name__ == '__main__':
    # load data
    pth = os.path.join(cfg.DIR_PROB, EXPERIMENT+'.txt')
    data = np.loadtxt(pth)
    N = len(data)
    prob = data[:,:51]
    y_pred = data[:,51].astype(np.int32)
    y_true = data[:,52].astype(np.int32)

    y_true_arr = np.zeros((N,51))
    for i in range(N): 
        y_true_arr[i,y_true[i]] = 1

    # analyze
    conf_matrix = confusion_matrix(y_true, y_pred)

    # plot
    '''
    plt.figure()
    ids = np.random.choice(N, N_SAMPLES, replace=False)
    plt.subplot(1,2,1)
    plt.imshow(prob[ids])
    plt.colorbar()
    plt.title('probability')
    plt.subplot(1,2,2)
    plt.imshow(truth[ids])
    plt.title('groundtruth')

    plt.figure()
    plt.plot(predict[ids], 'b-', label='prediction')
    plt.plot(lbl[ids], 'r-', label='truth')
    plt.legend()
    plt.title('top-1 prediction + groundtruth')
    '''
    plt.figure()
    plt.imshow(conf_matrix)
    plt.colorbar()
    plt.title('Confusion matrix')
    plt.show()
