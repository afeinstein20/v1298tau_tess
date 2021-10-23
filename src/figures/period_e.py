import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

COLOR = 'k'#'#FFFAF1'
plt.rcParams['font.size'] = 19
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.major.size']  = 8 
plt.rcParams['ytick.major.size']  = 8 

plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.minor.size']  = 6
plt.rcParams['ytick.minor.size']  = 6

plt.rcParams['axes.linewidth'] = 3
lw = 5

plt.rcParams['text.color'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['axes.edgecolor'] = COLOR
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['legend.facecolor'] = 'none'


summary = Table.read('../../data/trace_planete.csv', format='csv')

periods = np.arange(38,56,1,dtype=int)
keys = ['logl_{}'.format(i) for i in periods]
pkeys = ['period_{}'.format(i) for i in periods]

avg = 'logl'

fig = plt.figure(figsize=(8,5))

logl = summary[summary['col0']=='logl']['mean'].data

post_prob = np.zeros(len(keys))
means = np.zeros(len(keys))

for i in range(len(keys)):

    m1 = summary[summary['col0']==pkeys[i]]['mean'].data
    
    logl_p = summary[summary['col0']==keys[i]]['mean'].data
    pp = np.exp(logl_p - logl)
    post_prob[i] = pp
    means[i] = m1 + 0.0 

    plt.vlines(m1, -200, pp, lw=5, color='#742C64')


plt.vlines(24.141445*2, -200, 100, color='#1A48A0', linestyle='--', lw=3,
           label='2:1 resonance with V1298 Tau b = 48.28 Days')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", fontsize=16.5,
           borderaxespad=0.)
plt.xlabel('Median Period [Days]')
plt.ylabel('Posterior Probability')

plt.yscale('log')
plt.ylim(-20,0.01)
plt.savefig('periode.pdf', rasterize=True, bbox_inches='tight', dpi=250)
