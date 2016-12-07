import matplotlib.pyplot as plt
import numpy as np
import configure as cfg
import sys, os, ipdb
from sklearn.metrics import confusion_matrix


EXPERIMENT = 'fuseval_299'

def main(pth):
    # load data
    #pth = os.path.join(cfg.DIR_PROB, EXPERIMENT+'.txt')
    data = np.loadtxt(pth)
    N = len(data)
    n_classes = len(cfg.CLASSES)
    score = data[:,:n_classes]
    y_pred = data[:,n_classes].astype(np.int32)
    y_true = data[:,n_classes+1].astype(np.int32)

    y_true_arr = np.zeros((N,n_classes))
    for i in range(N): 
        y_true_arr[i,y_true[i]] = 1

    # analyze
    conf_matrix = confusion_matrix(y_true, y_pred)
    print 'prediction classes\n', np.unique(y_pred)
    print 'true classes\n', np.unique(y_true)
    c=0
    for i in range(N):
        if y_true[i]==y_pred[i]:
            c+=1
    print 'accuracy:', c*1.0/N

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
    plt.title('Confusion matrix: '+pth)


if __name__ == '__main__':
    for pth in sys.argv[1:]:
        main(pth)
    plt.show()
