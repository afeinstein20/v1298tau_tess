import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['axes.edgecolor'] = COLOR
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['legend.facecolor'] = 'none'

edgecolor = '#05021f'
k2_ec = '#b8b4b2'


path = '../../data/'
gp_mod = np.load(os.path.join(path, 'gp.npy'), allow_pickle=True).tolist()
map_soln=np.load(os.path.join(path,'map_soln.npy'), allow_pickle=True).tolist()
extras = np.load(os.path.join(path,'extras.npy'), allow_pickle=True).tolist()

planets=['c','d','b','e']
periods   = map_soln['period']
t0s = map_soln['t0']
t0s = np.append(t0s, [t0s[1]+periods[1], t0s[0]+periods[0], t0s[0]+periods[0]*2, 
                      t0s[1]+periods[1]*2, t0s[2]+periods[2], t0s[0]+periods[0]*3])
durations = np.array([4.66, 5.59, 6.42, 7.45, 5.59, 4.66, 4.66, 5.59, 6.42, 4.66])/24.0
parula = ['#eb9c3b', '#74BB43', '#1A48A0', '#742C64', '#74BB43', '#eb9c3b',
          '#eb9c3b', '#74BB43', '#1A48A0', '#eb9c3b']


## The data
time = gp_mod['time'] + 0.0
flux = gp_mod['flux'] + 0.0
flux_err = gp_mod['flux_err'] + 0.0
model = gp_mod['gp_mod'] + 0.0
planet_models = extras['light_curves_tess'] + 0.0

rest = np.array([planet_models[:,1], planet_models[:,0], 
                  planet_models[:,0], planet_models[:,1],
                  planet_models[:,2], planet_models[:,0]]).T
planet_models = np.append(planet_models, rest, axis=1)

####
#The Figure
####

fig3 = plt.figure(constrained_layout=True, figsize=(20,26))
fig3.set_facecolor('w')
gs = fig3.add_gridspec(4,4)

bigger =20

yflat = flux - model

## AX1
ax1 = fig3.add_subplot(gs[0, :])
ax1.set_title('Normalized & De-Trended TESS Light Curves for V1298 Tau', fontsize=26, y=1.3)

ax1.errorbar(time, flux, yerr=flux_err,
             color='w', marker='o', linestyle='',
             markeredgecolor=edgecolor, zorder=1,
             ecolor=edgecolor)#, capsize=3)

ax1.plot(time, model, lw=2.5, color='k', zorder=2,
         label='GP Model')
ax1.set_xlim(time[0], time[-1])
ax1.set_xticklabels([])

divider = make_axes_locatable(ax1)
rax = divider.append_axes("bottom", size='65%', pad=0)

ax1.set_ylabel('Flux [ppt]', fontsize=25, y=0.2)

rax.errorbar(time, yflat, yerr=flux_err,
             color='w', marker='o', linestyle='',
             markeredgecolor=edgecolor, zorder=1,
             ecolor=edgecolor)#, capsize=3)

rax.set_xlim(time[0], time[-1])
rax.set_xlabel('Time [BKJD - 2454833]', fontsize=25)


for i in range(len(t0s)):
    start = t0s[i]-durations[i]/2
    stop = t0s[i]+durations[i]/2
    if i < 4:
        ax1.axvspan(start, stop,
                    ymin=yflat.min(), ymax=yflat.max(), 
                    color=parula[i], lw=0, alpha=0.5, linestyle='',
                    label='Planet {}'.format(planets[i]))
    else:
        ax1.axvspan(start, stop,
                    ymin=-10,ymax=4, 
                    color=parula[i], lw=0, alpha=0.5, linestyle='')
        rax.axvspan(start, stop,
                    ymin=yflat.min(), ymax=yflat.max(), 
                    color=parula[i], lw=0, alpha=0.5, linestyle='')

ax1.legend(bbox_to_anchor=(0.1, 1.02, 1.3, .102), 
           loc='lower left',
           ncol=5, borderaxespad=0., 
           fontsize=20, markerscale=6)
rax.set_ylim(-9, 3.5)

transit_axes = []

widths = [0, [1,4], 
          [0,3], 3, 
          0, [1,4]]
line = [1, 1, 
        2, 2, 
        3, 3]
titles = ['Planet d', 'Planets b, c, & e', 
          'Planet d & c', 'Planet c', 
          'Planet d', 'Planets b & c']
occurrence = [1, [2, 0, 3], 
              [4, 5], 6, 
              7, [8,9]]

for i in range(len(widths)):
    if type(widths[i]) == list:
        ax3 = fig3.add_subplot(gs[line[i], widths[i][0]:widths[i][1]])
    else:
        ax3 = fig3.add_subplot(gs[line[i], widths[i]])
    divider = make_axes_locatable(ax3)
    rax3 = divider.append_axes("bottom", size='65%', pad=0)
        
    ax3.set_title(titles[i], fontsize=25)
    
    if type(occurrence[i]) == list:
        q = ( (time > t0s[occurrence[i][0]] - 0.5) &
              (time < t0s[occurrence[i][-1]] + 0.5) )
        if i == 2:
            q = ( (time > t0s[occurrence[i][-1]] - 0.5) &
                  (time < t0s[occurrence[i][0]] + 0.5) )

        for j in occurrence[i]:
            
            if i == 1 and j == 0:
                rax3.plot(time, 
                          np.nansum(planet_models[:,occurrence[i]], axis=1),
                         c='k', lw=4, zorder=3)
            rax3.plot(time, planet_models[:,j],
                      c=parula[j], lw=4, zorder=5)
            
            start = t0s[j]-durations[j]/2
            stop = t0s[j]+durations[j]/2

            ax3.axvspan(start, stop,
                        ymin=flux.min(), ymax=flux.max(), 
                        color=parula[j], lw=0, alpha=0.5, linestyle='')
    else:
        q = ( (time > t0s[occurrence[i]] - 0.5) &
              (time < t0s[occurrence[i]] + 0.5) )
        start = t0s[occurrence[i]]-durations[occurrence[i]]/2
        stop = t0s[occurrence[i]]+durations[occurrence[i]]/2

        ax3.axvspan(start, stop,
                    ymin=flux.min(), ymax=flux.max(), 
                    color=parula[occurrence[i]], lw=0, alpha=0.5, linestyle='')
        rax3.plot(time, planet_models[:,occurrence[i]],
                 c=parula[occurrence[i]], lw=4, zorder=5)
    
    ax3.errorbar(time[q], flux[q], 
                 yerr=flux_err[q],
                 color='w', marker='o', linestyle='',
                 markeredgecolor=edgecolor, zorder=3,
                 ecolor=edgecolor)
    rax3.errorbar(time[q], yflat[q], 
                  yerr=flux_err[q],
                  color='w', marker='o', linestyle='',
                  markeredgecolor=edgecolor, zorder=3,
                  ecolor=edgecolor)
    ax3.plot(time[q], model[q], c='k', lw=3, zorder=4)
    
    yticks = np.round([np.nanmin(flux[q]),  
                       (np.nanmin(flux[q]) +np.nanmean(flux[q]))/2.0,
                       np.nanmax(flux[q])], 1)
    ax3.set_yticks(yticks)
    
    yticks = np.round([np.nanmin(yflat[q]),  
                       (np.nanmin(yflat[q]) +np.nanpercentile(yflat[q], 95))/2.0,
                       np.nanpercentile(yflat[q], 95)], 1)
    rax3.set_yticks(yticks)
    rax3.set_ylim(np.nanmin(yflat[q]), np.nanpercentile(yflat[q],99))
    
    if i == 0 or i ==4 or i == 3:
        if i != 3:
            ax3.set_ylabel('Flux [ppt]', fontsize=25, y=0.02)
            
        ticks = np.round([time[q][5],  
                          time[q][int(len(time[q])/2)],
                          time[q][-5]],2)

        ax3.set_xticks(ticks)
        ax3.set_xticklabels([str(e) for e in ticks])
        rax3.set_xticks(ticks)
        rax3.set_xticklabels([str(e) for e in ticks])
        ax3.set_xlim(time[q][0], time[q][-1])
        rax3.set_xlim(time[q][0], time[q][-1])
    else:
        ax3.set_xlim(time[q][5], time[q][-5])
        rax3.set_xlim(time[q][5], time[q][-5])
        
    if i == 2:
        ax3.set_ylabel('Flux [ppt]', fontsize=25, y=0.2)
    if i > 3:
        rax3.set_xlabel('Time [BKJD - 2454833]', fontsize=25)

    
    transit_axes.append(ax3)
    transit_axes.append(rax3)
    ax3.set_xticklabels([])
    

all_axes = transit_axes
all_axes.append(ax1)
for ax in all_axes:
    ax.set_rasterized(True)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
plt.savefig('transits.pdf', rasterize=True, bbox_inches='tight', dpi=300)
