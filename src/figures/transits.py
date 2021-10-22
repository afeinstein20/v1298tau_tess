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

parula = ['#eb9c3b', '#74BB43', '#1A48A0', '#742C64']

planets=['c','d','b','e']
periods   = np.array([8.249147, 12.401369,  24.141445, 36.695032307689445])    
t0s       = np.array([4648.53, 4645.4, 4648.1, 4648.8, 
                      4645.4+periods[1], 4648.53+periods[0], 4648.53+periods[0]*2])
durations = np.array([4.66, 5.59, 6.42, 7.45, 5.59, 4.66, 4.66])/24.0


path = '../../data/'
gp_mod = np.load(os.path.join(path, 'gp.npy'), allow_pickle=True).tolist()
map_soln=np.load(os.path.join(path,'map_soln.npy'), allow_pickle=True).tolist()
extras = np.load(os.path.join(path,'extras.npy'), allow_pickle=True).tolist()

time = gp_mod['time'] + 0.0
flux = gp_mod['flux'] + 0.0
flux_err = gp_mod['flux_err'] + 0.0
model = gp_mod['gp_mod'] + 0.0
planet_models = extras['light_curves_tess'] + 0.0


fig3 = plt.figure(constrained_layout=True, figsize=(16,20))
fig3.set_facecolor('w')
gs = fig3.add_gridspec(6,4, height_ratios=[2,1,1,2,1.5,1.5])

bigger =20

## AX1
ax1 = fig3.add_subplot(gs[0, :])
ax1.set_title('TESS Light Curve for V1298 Tau', fontsize=26, y=1.2)
ax1.set_ylabel('Normalized Flux', fontsize=bigger)

ax1.errorbar(time, flux, yerr=flux_err,
             color='w', marker='o', linestyle='',
             markeredgecolor=edgecolor, zorder=1,
             ecolor=edgecolor)#, capsize=3)

ax1.plot(time, model, lw=2, color='#0fa0cc', zorder=2,
         label='GP Model')
ax1.set_xlim(time[0], time[-1])


## AX2
ax2 = fig3.add_subplot(gs[3, :])
ax2.set_title('Detrended Light Curve', fontsize=bigger)
ax2.set_ylabel('Normalized Flux', fontsize=bigger)
ax2.set_xlabel('Time [BJD - 2457000]', fontsize=bigger)

yflat =  flux-model#-0.5
ax2.errorbar(time, yflat, yerr=flux_err, 
             color='w', marker='o', linestyle='',
                 markeredgecolor=edgecolor, zorder=1,
                 ecolor=edgecolor)#, capsize=3)

for i in range(4):
    for j in range(3):
        start = t0s[i]+(periods[i]*j)-durations[i]/2
        stop = t0s[i]+(periods[i]*j)+durations[i]/2
        ax2.axvspan(start, stop,
                    ymin=flux.min(), ymax=flux.max(), 
                    color=parula[i], lw=0, alpha=0.5, linestyle='')
        if j == 1:
            ax1.axvspan(start, stop,
                        ymin=yflat.min(), ymax=yflat.max(), 
                        color=parula[i], lw=0, alpha=0.5, linestyle='',
                        label='Planet {}'.format(planets[i]))
        else:
            ax1.axvspan(start, stop,
                        ymin=yflat.min(), ymax=yflat.max(), 
                        color=parula[i], lw=0, alpha=0.5, linestyle='')

ax1.legend(bbox_to_anchor=(0.08, 1.02, 1.3, .102), loc='lower left',
           ncol=5, borderaxespad=0., fontsize=20, markerscale=6)#, mode="center", fontsize=20)

ax2.set_xlim(time[0], time[-1])

transit_axes = []

widths = [0, [1,4]]
titles = ['Planet d', 'Planets b, c, & e']
occurrence = [1, [2,0, 3]]

for i in range(len(widths)):
    if type(widths[i]) == list:
        ax3 = fig3.add_subplot(gs[4, widths[i][0]:widths[i][1]])
        ax5 = fig3.add_subplot(gs[1, widths[i][0]:widths[i][1]])
    else:
        ax3 = fig3.add_subplot(gs[4, widths[i]])
        ax5 = fig3.add_subplot(gs[1, widths[i]])
    ax3.set_title(titles[i], fontsize=bigger)
    ax5.set_title(titles[i], fontsize=bigger)
    
    if type(occurrence[i]) == list:
        q = ( (time > t0s[occurrence[i][0]] - 0.5) &
              (time < t0s[occurrence[i][-1]] + 0.5) )
        
        for j in occurrence[i]:
            ax3.plot(time, planet_models[:,j],
                     c=parula[j], lw=6, zorder=3)
            start = t0s[j]-durations[j]/2
            stop = t0s[j]+durations[j]/2

            ax5.axvspan(start, stop,
                        ymin=flux.min(), ymax=flux.max(), 
                        color=parula[j], lw=0, alpha=0.5, linestyle='')
    else:
        q = ( (time > t0s[occurrence[i]] - 0.5) &
              (time < t0s[occurrence[i]] + 0.5) )

        ax3.plot(time, planet_models[:,occurrence[i]],
             c=parula[occurrence[i]], lw=lw, zorder=3)
        
        start = t0s[occurrence[i]]-durations[occurrence[i]]/2
        stop = t0s[occurrence[i]]+durations[occurrence[i]]/2

        ax5.axvspan(start, stop,
                    ymin=flux.min(), ymax=flux.max(), 
                    color=parula[occurrence[i]], lw=0, alpha=0.5, linestyle='')
    
    ax3.errorbar(time[q], yflat[q], 
                 yerr=flux_err[q],
                 color='w', marker='o', linestyle='',
                 markeredgecolor=edgecolor, zorder=1,
                 ecolor=edgecolor)
    ax5.errorbar(time[q], flux[q], 
                 yerr=flux_err[q],
                 color='w', marker='o', linestyle='',
                 markeredgecolor=edgecolor, zorder=1,
                 ecolor=edgecolor)
    ax5.plot(time[q], model[q], c='#0fa0cc', lw=3)
    
    
    if i == 0:
        ax3.set_ylabel('De-trended flux [ppt]', fontsize=bigger)
        ticks = np.round([time[q][1],  time[q][int(len(time[q])/2)],
                          time[q][-2]],2)
        ax3.set_xticks(ticks)
        ax3.set_xticklabels([str(e) for e in ticks])
        ax5.set_xticks(ticks)
        ax5.set_xticklabels([str(e) for e in ticks])

    ax3.set_xlim(time[q][0], 
                 time[q][-1])
    ax5.set_xlim(time[q][0], 
                 time[q][-1])
    transit_axes.append(ax3)
    transit_axes.append(ax5)
    
    
ax4 = [fig3.add_subplot(gs[5, 0:3]), fig3.add_subplot(gs[5, 3])]
ax6 = [fig3.add_subplot(gs[2, 0:3]), fig3.add_subplot(gs[2, 3])]
titles = ['Planets d & c', 'Planet c']
occurrence = [[5,4], 6]

for i in range(2):
    ax4[i].set_title(titles[i], fontsize=bigger)
    ax6[i].set_title(titles[i], fontsize=bigger)
    
    if i == 0:
        ax4[i].set_ylabel('De-trended flux [ppt]')#, fontsize=bigger)
        ax6[i].set_ylabel('Normalized flux [ppt]', y=1.2)#, fontsize=bigger, y=1.2)
        
    if type(occurrence[i]) == list:
        q = ( (time > t0s[occurrence[i][0]] - 0.5) &
              (time < t0s[occurrence[i][-1]] + 0.5) )
        
        for j in occurrence[i]:
            ax4[i].plot(time, planet_models[:,j-4],
                        c=parula[j-4], lw=6, zorder=3)
            start = t0s[j-4]+periods[j-4]-durations[j-4]/2
            stop = t0s[j-4]+periods[j-4]+durations[j-4]/2

            ax6[i].axvspan(start, stop,
                        ymin=flux.min(), ymax=flux.max(), 
                        color=parula[j-4], lw=0, alpha=0.5, linestyle='')
            
    else:
        q = ( (time > t0s[occurrence[i]] - 0.5) &
              (time < t0s[occurrence[i]] + 0.5) )

        ax4[i].plot(time, planet_models[:,occurrence[i]-6],
                    c=parula[occurrence[i]-6], lw=lw, zorder=3)
        
        start = t0s[occurrence[i]-6]+periods[occurrence[i]-6]*2-durations[occurrence[i]-6]/2
        stop = t0s[occurrence[i]-6]+periods[occurrence[i]-6]*2+durations[occurrence[i]-6]/2

        ax6[i].axvspan(start, stop,
                        ymin=flux.min(), ymax=flux.max(), 
                        color=parula[occurrence[i]-6], lw=0, alpha=0.5, linestyle='')
        
    ax4[i].errorbar(time[q], yflat[q], 
                     yerr=flux_err[q],
                     color='w', marker='o', linestyle='',
                     markeredgecolor=edgecolor, zorder=1,
                     ecolor=edgecolor)#, capsize=3)
    ax6[i].errorbar(time[q], flux[q], 
                     yerr=flux_err[q],
                     color='w', marker='o', linestyle='',
                     markeredgecolor=edgecolor, zorder=1,
                     ecolor=edgecolor)#, capsize=3)
    ax6[i].plot(time[q], model[q], c='#0fa0cc', lw=3)
        
    ax4[i].set_xlabel('Time [BJD - 2457000]')
    ax4[i].set_xlim(time[q][0], 
                        time[q][-1])
    ax6[i].set_xlabel('Time [BJD - 2457000]')
    ax6[i].set_xlim(time[q][0], 
                        time[q][-1])
    
    if i == 1:
        ax4[i].set_xticks(np.round([time[q][1], 
                                    time[q][int(len(time[q])/2)],
                                    time[q][-1]],1))
        ax6[i].set_xticks(np.round([time[q][1], 
                                    time[q][int(len(time[q])/2)],
                                    time[q][-1]],1))
    transit_axes.append(ax4[i])
    transit_axes.append(ax6[i])
    
all_axes = transit_axes
all_axes.append(ax1)
all_axes.append(ax2)
for ax in all_axes:
    ax.set_rasterized(True)
    
plt.savefig('transits.pdf', rasterize=True, bbox_inches='tight', dpi=300)
      
