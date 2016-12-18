"""
References:
    https://gist.github.com/awjuliani/b596e9c3cac162905b22#file-t-sne-tutorial-ipynb
    https://www.tensorflow.org/versions/master/how_tos/embedding_viz/
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys, ipdb

subsampling = 5
logdir = '.'


def plot_2d(lowDWeights, labels):
    assert lowDWeights.shape[0] >= len(labels), "More labels than weights"
    plt.figure()
    for i, label in enumerate(labels):
        x, y = lowDWeights[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')
    plt.show()
    return


def plot_3d(lowDWeights, labels):
    from tensorflow.contrib.tensorboard.plugins import projector

    with tf.Graph().as_default():
        # prepare variables and environment
        sess = tf.Session()
        embedding_var = tf.Variable(lowDWeights, name='data')
        summary_writer = tf.train.SummaryWriter(logdir)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        # associate data with embeddings
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        # embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

        projector.visualize_embeddings(summary_writer, config)
        
        summary_writer.flush()
    return


if __name__ == '__main__':
    path = sys.argv[1]
    data = np.loadtxt(path) 

    rpst = data[::subsampling,:51] # representation of network's output
    labels = data[::subsampling,-1]


    tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
    lowDWeights = tsne.fit_transform(rpst)

    # plot as 2D
    plot_2d(lowDWeights, labels)
    
    # plot as 3D
    #plot_3d(lowDWeights, labels)
