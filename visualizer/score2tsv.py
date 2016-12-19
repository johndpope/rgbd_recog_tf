import numpy as np
"""
example:
    python score2tsv.py ../results/trial1_withbk/score/rgbeval_36.txt ../lists/train_1.lst
"""

import sys, ipdb


if __name__ == '__main__':
    score_path = sys.argv[1]
    meta_path = sys.argv[2]

    score = np.loadtxt(score_path)
    lst = open(meta_path).read().splitlines()
    score = score[:,:51]
    np.savetxt('data.tsv', score, delimiter='\t', fmt='%.4f')

    f = open('metadata.tsv', 'w')
    f.write('Category\tObject\tSample\n')
    for i in lst:
        toks = i.split('/')
        f.write('%s\t%s\t%s\n' % (toks[0], toks[1], toks[2]))
    f.close()

