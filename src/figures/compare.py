import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from lightkurve.lightcurve import LightCurve as LC
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import batman

COLOR = 'k'#'#FFFAF1'
plt.rcParams['font.size'] = 18
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

bigger = 24

def setup_batman(time, args, mission='K2', index=0):
    params = batman.TransitParams()
    if mission == 'K2':
        params.t0 = args[0]                     #time of inferior conjunction
        params.per = args[1]                        #orbital period
    elif mission=='phase':
        params.t0 = 0 + 0. 
        params.per = args[1]
    else:
        params.t0 = t0s[index]
        params.per= periods[index]
    params.rp = args[2]                        #planet radius (in units of stellar radii)
    params.a = args[3]                        #semi-major axis (in units of stellar radii)
    params.inc = args[4]                       #orbital inclination (in degrees)
    params.ecc = args[5]                        #eccentricity
    params.w = args[6]                         #longitude of periastron (in degrees)
    params.u = [0.46, 0.11]                #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       #limb darkening model
    
    m = batman.TransitModel(params, time)    #initializes model
    flux = m.light_curve(params)          #calculates light curve
    return (flux*1e3)-1e3


edgecolor = '#05021f'
k2_ec = '#b8b4b2'
parula = ['#eb9c3b', '#74BB43', '#1A48A0', '#742C64']

## LOADS IN GP FIT FOR TRANSIT MODELING
gp_mod = np.load('../../data/gp.npy', allow_pickle=True).tolist()
map_soln=np.load('../../data/map_soln.npy', allow_pickle=True).tolist()
extras = np.load('../../data/extras.npy', allow_pickle=True).tolist()

time = gp_mod['time'] + 0.0
flux = gp_mod['flux'] + 0.0
flux_err = gp_mod['flux_err'] + 0.0
model = gp_mod['gp_mod'] + 0.0
planet_models = extras['light_curves_tess'] + 0.0

planets=['c','d','b','e']
periods   = map_soln['period']
t0s = map_soln['t0']
t0s = np.append(t0s, [t0s[1]+periods[1], t0s[0]+periods[0], t0s[0]+periods[0]*2])
durations = np.array([4.66, 5.59, 6.42, 7.45, 5.59, 4.66, 4.66])/24.0

planets=['c','d','b','e']
durations = np.array([4.66, 5.59, 6.42, 7.45])/24.0

## LOADS IN INFORMATION FROM K2 DATA
t0s_k2   = np.array([2231.281202, 2239.400529, 2234.046461, 2263.6229])
periods_k2 = np.array([8.249147, 12.401369,  24.141445, 36.695032307689445])

c = [2231.281202, 8.249147, 0.0381, 13.19, 88.49, 0, 92]
d = [2239.400529, 12.401369,  0.0436, 17.31, 89.04, 0, 88]
b = [2234.046461, 24.141445, 0.07, 27.0, 89, 0, 85]
e = [2263.6229, 60, 0.0611, 51, 89.4, 0, 91]

batman_params=[c,d,b,e]

ror = np.load('../../data/results.npy')

## SETS UP THE FIGURE
fig = plt.figure(constrained_layout=True, figsize=(12,18))
fig.set_facecolor('w')
gs = fig.add_gridspec(3,2, height_ratios=[2,2,1.5])

plt.rcParams['font.size'] = 18
bigger=22

fig.set_facecolor('w')
axes = [fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]), 
        fig.add_subplot(gs[1, 0]), 
        fig.add_subplot(gs[1, 1])]
last = fig.add_subplot(gs[2, :])

k2_ec = '#8a8988'

for i in range(4):
    axes[i].set_title('V1298 Tau {}'.format(planets[i]), fontsize=bigger)
    
    divider = make_axes_locatable(axes[i])
    rax = divider.append_axes("bottom", size='30%', pad=0)
    axes[i].figure.add_axes(rax)
    
    mask_inds = np.delete(np.arange(0,4,1), i)
    q = np.nansum(planet_models[:,mask_inds], axis=1) >= 0
    
    lk = LC(time=time[q], 
            flux=flux[q]-model[q],
            flux_err=flux_err[q]).fold(epoch_time=map_soln['t0'][i], 
                                       period=map_soln['period'][i])
    
    md = LC(time=time[q], 
                flux=planet_models[:,i][q],
                flux_err=flux_err[q]).fold(epoch_time=map_soln['t0'][i], 
                                           period=map_soln['period'][i])
    
    axes[i].errorbar(lk.time.value,
                     lk.flux.value, yerr=lk.flux_err.value,
                     marker='o', color='w', markeredgecolor=k2_ec,
                     zorder=1, label='TESS', ecolor=k2_ec, linestyle='')
    
    axes[i].plot(md.time.value,
                 md.flux.value, c=parula[i],
                 zorder=3, lw=lw)

    
    phase = np.linspace(-40,40,len(lk.time.value))
    k2_model = setup_batman(phase, batman_params[i], mission='phase')
    interp = interp1d(phase, k2_model)
    lkk2 = LC(time=lk.time.value,
              flux=interp(lk.time.value))
    
    axes[i].plot(lkk2.time.value,
                 lkk2.flux.value,
                 color='k',
                 zorder=2, label='K2', lw=lw)

    axes[i].set_xticks(np.arange(-1,1.5,0.5))
    rax.set_xticks(np.arange(-1,1.5,0.5))
    axes[i].set_xticklabels([])
    
    rax.plot(lk.time.value, 
             lk.flux.value-md.flux.value, 'wo', 
             markeredgecolor=k2_ec)#, zorder=2)
    
    rax.plot(lk.time.value, 
             lk.flux.value-lkk2.flux.value, 'ko', zorder=1)
    
    rax.set_xticks(np.arange(-1,1.5,0.5))
    rax.set_ylim(-3,3)
    
    if i == 0:
        axes[i].legend(bbox_to_anchor=(0.7, 1.15, 0.7, .102), loc='lower left',
                       ncol=2, mode="expand", borderaxespad=0., fontsize=bigger)
        
    if i == 0 or i == 2:
        axes[i].set_ylabel('De-trended Flux [ppt]')
        rax.set_ylabel('Residuals')
    if i >= 2:
        rax.set_xlabel('Time from Mid-Transit [days]', fontsize=bigger)
    
    
    axes[i].set_xlim(-1,1)
    rax.set_xlim(-1,1)
    axes[i].set_ylim(-7.5,3)
    axes[i].set_rasterized(True)


k2errs = [0.0017, 0.0022, 0.0023, 0.004]
rprs = [0.0381, 0.0436, 0.0700, 0.0611]
lims = [0.032, 0.075]

for i, rp in enumerate(rprs):
    last.errorbar(rp, 
                  np.nanmedian(ror[i]),
                  xerr=k2errs[i],
                  yerr=np.nanstd(ror[i]),
                  marker='o', color=parula[i], ms=10)
last.plot(np.linspace(0,20,10), np.linspace(0,20,10), 'k')

last.set_xlim(lims)
last.set_ylim(lims)

last.set_xlabel('K2 $R_p/R_{star}$', fontsize=bigger)
last.set_ylabel('TESS $R_p/R_{star}$', fontsize=bigger)
    
plt.subplots_adjust(hspace=0.3)
plt.savefig('paper/folded_compare.pdf', rasterize=True, dpi=250,
            bbox_inches='tight')
