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
    print 'Prediction classes\n', np.unique(y_pred)
    print 'True classes\n', np.unique(y_true)

    for t in range(5):
        count = 0
        for i in range(N):
            foo = score[i]
            if y_true[i] in foo.argsort()[-(t+1):][::-1]: count+=1
        print 'Top', t+1, 'results:', count*1.0/N
    '''
    t1=0
    for i in range(N):
        if y_true[i]==y_pred[i]: t1+=1
    print 'Top 1 accuracy:', t1*1.0/N


    t5=0
    for i in range(N):
        foo = score[i]
        if y_true[i] in foo.argsort()[-5:][::-1]: t5+=1
    print 'Top 5 accuracy:', t5*1.0/N
    '''

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
