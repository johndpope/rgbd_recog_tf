import os, glob, ipdb
import configure as cfg
import numpy as np


def make_lists(category, objs, parrent_path, file):
    class_pth = os.path.join(parrent_path, category)
    for obj in objs:
        obj_pth = os.path.join(class_pth, obj)
        obj_pth_out = os.path.join(cfg.DIR_DATA, category, obj)

        sample_ids = glob.glob1(obj_pth, '*_loc.txt') # samples
        sample_ids = [i.replace('_loc.txt','') for i in sample_ids]
        sample_ids.sort()

        for sid in sample_ids:
            line = os.path.join(category, obj, sid) + '\n'
            file.write(line)
    return


'''
def main():
    train_f = open(cfg.PTH_TRAIN_LST, 'w')
    eval_f = open(cfg.PTH_EVAL_LST, 'w')
    test_f = open(cfg.PTH_TEST_LST, 'w')

    # go through the whole dataset
    classes = os.listdir(cfg.DIR_DATA_RAW) # classes
    if '.DS_Store' in classes: classes.remove('.DS_Store')
    if not os.path.exists(cfg.DIR_DATA): os.mkdir(cfg.DIR_DATA)

    for a_class in classes:
        class_pth = os.path.join(cfg.DIR_DATA_RAW, a_class)
        objs = os.listdir(class_pth) # objects
        if '.DS_Store' in objs: objs.remove('.DS_Store')

        train_objs = objs[:-2]
        eval_objs = [objs[-2]]
        test_objs = [objs[-1]]

        make_lists(a_class, train_objs, train_f)
        make_lists(a_class, eval_objs, eval_f)
        make_lists(a_class, test_objs, test_f)

    train_f.close()
    eval_f.close()
    test_f.close()
    return
'''


def make_full_list(data_dir, outfile):
    f = open(outfile, 'w')

    # go through the whole dataset
    categories = open(cfg.PTH_DICT).read().splitlines()
    for category in categories:
        class_pth = os.path.join(data_dir, category)
        objs = os.listdir(class_pth) # objects
        objs = [o for o in objs if not o.startswith('.')]
        objs.sort()
        make_lists(category, objs, data_dir, f)

    f.close()
    return


def make_train_list(trial_splits, data_dir, out_paths):
    for trial in range(cfg.N_TRIALS):
        train_f = open(out_paths[trial], 'w')
        objs = trial_splits[trial][0]
        for obj in objs:
            category = obj[:obj.rindex('_')]
            make_lists(category, [obj], data_dir, train_f)
        train_f.close()
    return


def make_eval_list():
    testinstance_f = open(cfg.PTH_TESTINSTANCE_IDS, 'r')
    testinstance_lines = testinstance_f.read().splitlines()

    trial = 0
    for line in testinstance_lines:
        # begin new trial
        if line.startswith('******'):
            eval_f = open(cfg.PTH_EVAL_LST[trial], 'w')
            trial += 1
            continue

        # end trial
        if line == '':
            if not eval_f.closed:
                eval_f.close()
            continue

        a_class = line[:line.rindex('_')]
        objs = [line]
        make_lists(a_class, objs, cfg.DIR_DATA_EVAL_RAW, eval_f)

    testinstance_f.close()
    return


def make_trial_split():
    # retrievw all objects ids
    categories = open(cfg.PTH_DICT, 'r').read().splitlines()
    obj_ids = []
    for category in categories:
        foo = os.listdir(os.path.join(cfg.DIR_DATA_EVAL_RAW, category))
        obj_ids += [bar for bar in foo if not bar.startswith('.')]

    #
    trial_lines = open(cfg.PTH_TESTINSTANCE_IDS, 'r').read().splitlines()
    trial = 0
    trial_splits = [None]*cfg.N_TRIALS
    for line in trial_lines:
        # begin new trial
        if line.startswith('******'):
            eval_objs = []
            train_objs = []
            continue

        # end trial
        if line == '':
            if not train_objs == []:
                continue
            train_objs = [foo for foo in obj_ids if foo not in eval_objs]
            train_objs.sort(); eval_objs.sort()
            trial_splits[trial] = (train_objs, eval_objs)
            trial += 1
            continue

        eval_objs += [line]
    np.save(cfg.PTH_TRIAL_SPLIT, trial_splits)
    return trial_splits


if __name__ == '__main__':
    #make_full_list(cfg.DIR_DATA_RAW, cfg.PTH_FULL_LST)
    #make_full_list(cfg.DIR_DATA_EVAL_RAW, cfg.PTH_FULLEVAL_LST)

    trial_splits = make_trial_split()
    make_train_list(trial_splits, cfg.DIR_DATA_EVAL_RAW, cfg.PTH_TRAIN_SHORT_LST)
    make_train_list(trial_splits, cfg.DIR_DATA_RAW, cfg.PTH_TRAIN_LST)
    make_eval_list()
