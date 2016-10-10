import tensorflow as tf
import numpy as np

def batch_input(dataset, batch_size, train, num_preprocess_threads=None, num_readers=1):
    """Contruct batches of training or evaluation examples from the image dataset.

    Args:
        dataset: instance of Dataset class specifying the dataset.
            See dataset.py for details.
        batch_size: integer
        train: boolean
        num_preprocess_threads: integer, total number of preprocessing threads
        num_readers: integer, number of parallel readers

    Returns:
        images: 4-D float Tensor of a batch of images
        labels: 1-D integer Tensor of [batch_size].
    
    Raises:
        ValueError: if data is not found
    """

N = 26
batch_size = 2
lbl_lst = np.arange(1,N+1) 
dat_lst = np.array([chr(i+ord('a')-1) for i in lbl_lst])

print lbl_lst, dat_lst
