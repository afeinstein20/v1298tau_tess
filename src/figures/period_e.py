import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from astropy.table import Table
from scipy.interpolate import interp1d
from lmfit import Model

COLOR = 'k'#'#FFFAF1'
plt.rcParams['font.size'] = 16
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.major.size']  = 8 #12
plt.rcParams['ytick.major.size']  = 8 #12

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
#plt.rcParams['axes.spines.top'] = False
#plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['axes.edgecolor'] = COLOR
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['legend.facecolor'] = 'none'


summary0 = Table.read('trace_planete0.csv', format='csv')
summary1 = Table.read('trace_planete1.csv', format='csv')
summary2 = Table.read('trace_planet22.csv', format='csv')

periods = np.arange(44,56,1,dtype=int)
keys = ['logl_{}'.format(i) for i in periods]

fig = plt.figure(figsize=(8,5))
summaries = [summary0, summary1, summary2]

x,y=np.array([]),np.array([])
yerr=np.array([])

for i in range(len(keys)):

    for j,s in enumerate(summaries):
        m1 = s[s['col0']=='period_{}'.format(periods[i])]['mean'].data
        logl = s[s['col0']=='logl']['mean'].data
        logl_p = s[s['col0']==keys[i]]['mean'].data
        pp = np.nanmean(np.exp(logl_p - logl))

        plt.vlines(m1, -200, pp, lw=5, color='#742C64')


plt.vlines(24.141445*2, -200, 100, color='#1A48A0', linestyle='--', lw=3,
           label='2:1 resonance with V1298 Tau b')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, #mode="expand", 
           borderaxespad=0.)
plt.xlabel('Median Period [Days]')
plt.ylabel('Posterior Probability')

plt.yscale('log')
plt.ylim(-3,0.1)
plt.savefig('periode.pdf', rasterize=True, bbox_inches='tight', dpi=250)
