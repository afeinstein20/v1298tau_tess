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


summary = Table.read('../../data/planet_e_posteriors.tab', format='csv')

plt.figure(figsize=(7,4))

for i in range(len(p)):
    plt.vlines(summary['period'][i], 0, 
               np.abs(summary['posterior_prob'][i]), lw=5, color='#742C64')
plt.vlines(24.1382*2, 0,1, linestyle='--', color=parula[60], lw=3,
           label='2:1 resonance with V1298 Tau b = {} days'.format(np.round(24.1382*2,2)))
plt.ylim(0.03,0.075)
plt.xlabel('Median Period [days]', fontsize=18)
plt.ylabel('Posterior Probability', fontsize=18)
plt.legend(fontsize=14,bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.xlim(43,62)
plt.savefig('periode.pdf', 
            bbox_inches='tight', dpi=250,
            rasterize=True)
