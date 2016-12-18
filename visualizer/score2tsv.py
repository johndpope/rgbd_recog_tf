import numpy as np
import sys, ipdb

lst = open('../lists/eval_1.lst').read().splitlines() #TODO

if __name__ == '__main__':
    score_path = sys.argv[1]
    score = np.loadtxt(score_path)
    score = score[:,:51]
    np.savetxt('data.tsv', score, delimiter='\t', fmt='%.4f')

    f = open('metadata.tsv', 'w')
    f.write('Category\tObject\tSample\n')
    for i in lst:
        toks = i.split('/')
        f.write('%s\t%s\n' % (toks[0], toks[1], toks[2]))
    f.close()

