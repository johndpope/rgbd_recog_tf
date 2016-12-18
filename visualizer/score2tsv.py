import numpy as np
import ipdb

score = np.loadtxt('./rgbeval_49.txt')
lst = open('./eval_2.lst').read().splitlines()

score = score[:,:51]
np.savetxt('data.tsv', score, delimiter='\t', fmt='%.4f')

f = open('metadata.tsv', 'w')
f.write('Sample\tCategory\n')
for i in lst:
    toks = i.split('/')
    f.write('%s\t%s\n' % (toks[2], toks[0]))
f.close()

