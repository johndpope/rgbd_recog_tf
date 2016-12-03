import matplotlib.pyplot as plt
import numpy as np
import configure as cfg
import os, ipdb


EXPERIMENT = 'rgbeval_9'
N_SAMPLES = 150

if __name__ == '__main__':
    pth = os.path.join(cfg.DIR_PROB, EXPERIMENT+'.txt')
    data = np.loadtxt(pth)
    N = len(data)
    prob = data[:,:51]
    predict = data[:,51]
    lbl = data[:,52].astype(np.int32)
    truth = np.zeros((N,51));
    for i in range(N): truth[i,lbl[i]] = 1

    plt.figure()
    ids = np.random.choice(N, N_SAMPLES, replace=False)

    plt.subplot(1,2,1)
    plt.imshow(prob[ids])
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(truth[ids])
    plt.show()
