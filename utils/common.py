import os, sys, glob, ipdb
import numpy as np


def early_stopping():
    return


def get_sample_ids(dir_data, pth_lst):
    """Generate the paths to samples of a list
    Args:
        dir_data: where the data are stored
        pth_lst: list containing objects' names, e.g. training list, evaluation list, testing list
    Returns:
        sample_ids: paths of all sample ID that belong to the objects in pth_lst
    """

    def _get_sample_ids_of_obj(pth):
        """Get the indices (without extension) of samples in a directory.
        Args:
            pth: directory to search.
        Returns:
            idx_lst: list of indices without extensions
        """
        assert os.path.exists(pth), pth+' does not exist.'

        pwd = os.getcwd()
        os.chdir(pth)
        idx_lst = glob.glob('*_depth.png')
        idx_lst = ' '.join(idx_lst).replace('_depth.png', '').split()
        idx_lst.sort()
        os.chdir(pwd)
        return idx_lst

    objects = open(pth_lst, 'r').read().splitlines()
    sample_ids = []
    for obj in objects:
        pth = os.path.join(dir_data, obj)
        sample_ids +=  _get_sample_ids_of_obj(pth)
    return sample_ids


def get_paths_labels(dir_data, pth_lst, classes, to_shuffle):
    """Get the full paths and labels of samples
    Args:
        dir_data: directory containing data
        pth_lst: path to list
        classes: array of all possible classes
        to_shuffle: samples should be shuffled (for training) or not
    Returns:
        rgb_paths: paths to all rgb images
        dep_paths: paths to all depth images
        labels: list of labels, each row is a binary array, where 1 means the class the sample belongs to
    """
    sample_ids = get_sample_ids(dir_data, pth_lst)
    if to_shuffle:
        np.random.shuffle(sample_ids)

    rgb_paths = ['']*len(sample_ids)
    dep_paths = ['']*len(sample_ids)
    labels = np.zeros((len(sample_ids), len(classes)))
    for i in range(len(sample_ids)):
        tokens = sample_ids[i].split('_')
        L = len(tokens)
        class_id = '_'.join(tokens[:L-3])
        obj_id   = '_'.join(tokens[:L-2])

        rgb_paths[i] = os.path.join(dir_data, class_id, obj_id, sample_ids[i]+'.png')
        dep_paths[i] = os.path.join(dir_data, class_id, obj_id, sample_ids[i]+'_depth.png')
        labels[i, classes.index(class_id)] = 1
    return rgb_paths, dep_paths, labels


def load_images():
    return
