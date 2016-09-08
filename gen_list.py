import os, glob, ipdb
import configure as cfg


def make_lists(a_class, objs, file):
    class_pth = os.path.join(cfg.DIR_DATA_RAW, a_class)
    for obj in objs:
        obj_pth = os.path.join(class_pth, obj)
        obj_pth_out = os.path.join(cfg.DIR_DATA, a_class, obj)

        sample_ids = glob.glob1(obj_pth, '*_loc.txt') # samples
        sample_ids = [i.replace('_loc.txt','') for i in sample_ids]
        sample_ids.sort()

        for sid in sample_ids:
            line = os.path.join(a_class, obj, sid) + '\n'
            file.write(line)
    return


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
        eval_objs = objs[-2]
        test_objs = objs[-1]

        make_lists(a_class, train_objs, train_f)
        make_lists(a_class, eval_objs, eval_f)
        make_lists(a_class, test_objs, test_f)

    train_f.close()
    eval_f.close()
    test_f.close()
    return


if __name__ == '__main__':
    main()