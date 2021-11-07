import numpy as np
import matplotlib.pyplot as plt
from lightkurve.lightcurve import LightCurve

import matplotlib as mpl

from astropy.time import Time
from astropy import time, coordinates as coord, units as u
import astropy.constants as c

from mpl_toolkits.axes_grid1 import make_axes_locatable

import time as timer

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lightkurve.lightcurve import LightCurve as LC
from lightkurve.search import search_lightcurve
from mpl_toolkits.axes_grid1 import make_axes_locatable
import batman
from astropy.table import Table


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
#plt.rcParams['axes.spines.top'] = False
#plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['axes.edgecolor'] = COLOR
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['legend.facecolor'] = 'none'

edgecolor = '#05021f'
k2_ec = '#b8b4b2'

parula = np.load('/Users/arcticfox/parula_colors.npy')[np.linspace(0,160,4,dtype=int)]
parula = ['#eb9c3b', '#74BB43', '#1A48A0', '#742C64', '#74BB43', '#eb9c3b',
          '#eb9c3b', '#74BB43', '#1A48A0', '#eb9c3b']
#parula = ['#B3240B', '#74BB43', '#0494EC', '#BC84DC']

gp_mod = np.load('gp_loose.npy', allow_pickle=True).tolist()
map_soln=np.load('map_soln_loose.npy', allow_pickle=True).tolist()
extras = np.load('extras_loose.npy', allow_pickle=True).tolist()

planets=['c','d','b','e']
periods   = map_soln['period']#np.array([8.249147, 12.401369,  24.141445, 36.695032307689445])  
t0s = map_soln['t0']
t0s = np.append(t0s, [t0s[1]+periods[1], t0s[0]+periods[0], t0s[0]+periods[0]*2, 
                      t0s[1]+periods[1]*2, t0s[2]+periods[2], t0s[0]+periods[0]*3])
k2_t0 =  np.array([2231.2797, 2239.3913, 2234.0488, 2263.6229])
k2_pd =  np.array([8.24958, 12.4032, 24.1396, 60])
durations = np.array([4.66, 5.59, 6.42, 7.45, 5.59, 4.66, 4.66, 5.59, 6.42, 4.66])/24.0

import warnings
warnings.filterwarnings("ignore")

tangerine = ['#1a2754','#f9a919','#2ab59f','#e2721c','#db3e0e','#067bc1']

gp_mod = np.load('gp_loose.npy', allow_pickle=True).tolist()
time = gp_mod['time'] + 0.0

gp_mod = np.load('gp_loose.npy', allow_pickle=True).tolist()
map_soln=np.load('map_soln_loose.npy', allow_pickle=True).tolist()
extras = np.load('extras_loose.npy', allow_pickle=True).tolist()



planets=['c','d','b','e']
periods   = map_soln['period']#np.array([8.249147, 12.401369,  24.141445, 36.695032307689445])  
t0s = map_soln['t0']
t0s = np.append(t0s, [t0s[1]+periods[1], t0s[0]+periods[0], t0s[0]+periods[0]*2, 
                      t0s[1]+periods[1]*2, t0s[2]+periods[2], t0s[0]+periods[0]*3])
t0s = [t0s[2], t0s[2]+periods[2], t0s[3]]
durations = np.array([4.66, 5.59, 6.42, 7.45, 5.59, 4.66, 4.66, 5.59, 6.42, 4.66])/24.0


tpf1 = np.load('fluxes_o1.npy')
tpf2 = np.load('tpf.npy')
tpf3 = np.load('/Users/arcticfox/Downloads/tpf.npy')
all_tpfs = [tpf1, tpf2, tpf3]


for i in range(len(all_tpfs)):
    if all_tpfs[i].shape[2] != 26:
        if i == 0:
            all_tpfs[i] = all_tpfs[i][:,2:28, 2:28] + 0.0
        elif i == 2:
            all_tpfs[i] = all_tpfs[i][:,3:29, 3:29] + 0.0
    print(all_tpfs[i].shape)



fig, axes = plt.subplots(ncols=3, figsize=(14,6))
mask1 = np.zeros(all_tpfs[0][0].shape)
mask2 = np.zeros(all_tpfs[0][0].shape)
mask1[12:15,11:15]=1
mask2[13:16,12:16]=1
masks = [mask1, mask2, mask1]

total_len = 0

for i,ax in enumerate(axes.reshape(-1)):
    if i < 2:
        ax.imshow(all_tpfs[i][100], vmax=1e6, origin='lower')
    else:
        ax.imshow(np.rot90(all_tpfs[i][100],2), vmax=1e6, origin='lower')
    ax.imshow(masks[i], alpha=0.6, cmap='Greys')
    total_len += all_tpfs[i].shape[0]


start = 0
cutend = 28
lightcurves = np.zeros((total_len-cutend*3-3,3,4))
for n in range(len(all_tpfs)):
    x,y = np.where(masks[n]==1)
    x = np.unique(x)
    y = np.unique(y)
    for i in range(len(x)):
        for j in range(len(y)):
            f = all_tpfs[n][:len(all_tpfs[n])-cutend-1,x[i], y[j]]
            bad = np.where((f>np.nanmedian(f)+5*np.nanstd(f)) |
                           (f<np.nanmedian(f)-5*np.nanstd(f)))[0]
            f[bad] = np.nan
            lightcurves[start:start+len(f),i,j]= f/np.nanmedian(f) + 0.0
    start += len(f)


lc_dict = {}
keys = ['b1_{0}_{1}', 'b2_{0}_{1}', 'e_{0}_{1}']
t0s[0]+=0.03
t0s[1]+=0.09
t0s[2]+=0.03

for n,t in enumerate(t0s):
    for i in range(3):
        
        for j in range(4):
            q = ((time>=t-0.4) & (time<=t+0.4))
            lk = LightCurve(time=time[q], 
                           flux=(lightcurves[:,i,j][q]/np.nanmedian(lightcurves[:,i,j][q])-1)*1e3, 
                           flux_err=np.full(len(lightcurves[:,i,j][q]),1.5)).remove_outliers()
            lc_dict[keys[n].format(i,j)] = [np.ascontiguousarray(lk.time.value, dtype=np.float64),
                                            np.ascontiguousarray(lk.flux.value, dtype=np.float64),
                                            np.ascontiguousarray(lk.flux_err.value, dtype=np.float64)]


tess_texp = 0.006944432854652405
import batman
from lmfit.models import Model

keys = list(lc_dict.keys())


def setup_batman(time, rp, t0, a):
    global f, i
    endinds = np.arange(0,len(f)-1,1,dtype=int)
    endinds = np.delete(endinds, np.arange(40,90,1))
    if i >= 1:
        deg=4
    else:
        deg=2
    coeffs = np.polyfit(time[endinds], f[endinds], deg=deg)
    trend  = np.poly1d(coeffs)
    
    params = batman.TransitParams()
    params.t0 = t0                     #time of inferior conjunction
    params.per = 24.13702000215812                        #orbital period
    params.rp = rp                        #planet radius (in units of stellar radii)
    params.a = a
    params.inc = 89.0                       #orbital inclination (in degrees)
    params.ecc = 0.0                        #eccentricity
    params.w = 85.0                         #longitude of periastron (in degrees)
    params.u = [0.46, 0.11]                #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       #limb darkening model

    m = batman.TransitModel(params, time)    #initializes model
    flux = m.light_curve(params) # + trend(time)    #calculates light curve
    return (flux*1e3)-1e3 + trend(time)


all_output = []

for i in range(len(keys)):

    f = lc_dict[keys[i]][1] + 0.0

    if i < 9:
        tt = t0s[0]+0.0
    elif i < 18:
        tt = t0s[1]+0.0
    else:
        tt=t0s[2]+0.0
        
    tmodel = Model(setup_batman)
    pars = tmodel.make_params()
    pars['rp'].set(value=0.065, min=0.001, max=0.08)
    pars['t0'].set(value=tt, min=tt-0.3, max=tt+0.3)
    pars['a'].set(value=20.0, min=3.0, max=50.0)
    init = tmodel.eval(pars,time=lc_dict[keys[i]][0])
    output = tmodel.fit(f, pars, 
                        time=lc_dict[keys[i]][0],
                        weights=1.0/lc_dict[keys[i]][2])
    all_output.append([output, output.minimize(max_nfev=3000)])


grid = np.zeros((3,all_tpfs[0].shape[1], all_tpfs[0].shape[2]))

x,y = np.where(masks[0]==1)
x = np.unique(x)
y = np.unique(y)

z = 0
for g in range(len(grid)):
    for i in range(len(x)):
        for j in range(len(y)):
            mean = all_output[z][-1].params['rp'].value
            grid[g][x[i]][y[j]] = mean + 0.0
            z += 1

import os
import numpy as np
import requests
import matplotlib.ticker as ticker

from astroquery.mast import Tesscut
from astroquery.mast import Catalogs
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm

import lightkurve as lk

import matplotlib.pyplot as plt
from reproject import reproject_interp
import matplotlib.gridspec as gs
from matplotlib.colors import LogNorm

from IPython.display import Image
from ipywidgets import interact, interactive, fixed, interact_manual

import cv2


pixelsize = 11

# select a target and cutout size in pixels
ra, dec = 186.7679384635554, -72.4518042663561 
ra, dec = 61.331654, 20.157032

target_x = 217
target_y = 1940

target = '{0}, {1}'.format(ra, dec)
x = y = pixelsize ## pixel size
# set local file path to current working directory
path = os.path.abspath(os.path.curdir)

files = ['s0043/cam3-ccd2/hlsp_tica_tess_ffi_s0043-o1-00180084-cam3-ccd2_tess_v01_img.fits']
hdu = fits.open(files[0])
wcs = WCS(hdu[0].header)

n_pix = hdu[0].data.shape[0]
res = 21.0 * (u.arcmin/u.pixel) ## I think it's actually 20.98 or something. This is fine. 
area = res * (n_pix * u.pixel)
d = area.to(u.degree)
fov = d.value 


# compute the pixel area in arcmin
data = hdu[0].data
norm = simple_norm(data, 'log')
arcmin = area.to(u.arcmin).value
# retrieve the DSS file
filename = ['./dss_red_61.331654_20.157032.fits']#

dss = fits.open(filename[0])
# get the data and WCS for the DSS image
dss_data = dss[0].data
dss_wcs = WCS(dss[0].header)


# get RA and Dec coords of catalog data
from astropy.table import Table
catalogData = Table.read('catalogData.tab', format='ascii')
catalogData = catalogData[np.where(catalogData['Tmag']<14)]
tic_ra = catalogData['ra']
tic_dec = catalogData['dec']

# get pixel coordinates of RA and Dec 
coords = wcs.all_world2pix(list(zip(tic_ra, tic_dec)),0)
xc = np.array([c[0] for c in coords])
yc = np.array([c[1] for c in coords])

inds = np.where( (np.abs(np.diff(xc-target_x))<5) &
                 (np.abs(np.diff(yc-target_y))<5) )[0]
xc = xc[inds]
yc = yc[inds]


new_data = np.full(data.shape, np.nan)
newy, newx = target_y+2, target_x+2
new_data[newy-2:newy+1,newx-3:newx+1] = grid[0,12:15,11:15]#np.rot90(grid[2,12:15,11:15],2)


reproj_tesscut1, footprint = reproject_interp((new_data, wcs,), 
                                              dss_wcs, 
                                              shape_out=dss_data.shape, 
                                              order='nearest-neighbor')

new_data = np.full(data.shape, np.nan)
newy, newx = target_y+2, target_x+2
new_data[newy-2:newy+1,newx-3:newx+1] = grid[2,12:15,11:15]#np.rot90(grid[2,12:15,11:15],2)


reproj_tesscut2, footprint = reproject_interp((new_data, wcs,), 
                                              dss_wcs, 
                                              shape_out=dss_data.shape, 
                                              order='nearest-neighbor')


time = gp_mod['time'] + 0.0
flux = gp_mod['flux'] + 0.0
flux_err = gp_mod['flux_err'] + 0.0
model = gp_mod['gp_mod'] + 0.0
planet_models = extras['light_curves_tess'] + 0.0


k2_lc = Table.read('v1298tau-tjd-flat-20190805.csv',format='csv')
lc_k2 = LC(time=k2_lc['x'], flux=k2_lc['y'], flux_err=k2_lc['yerr'])


k2_planet_masks = np.zeros((4,len(k2_lc)))
for i in range(len(k2_pd)):
    pds = k2_t0[i] + k2_pd[i]*np.arange(0,10,1,dtype=int)
    for period in pds:
        inds = np.where((lc_k2.time.value>period-durations[i]/2.0) &
                        (lc_k2.time.value<period+durations[i]/2.0))[0]
        k2_planet_masks[i][inds] = 1


ror = np.load('results.npy')


for i in range(4):
    inds = np.delete(np.arange(0,4,1,dtype=int),i)
    mask = ((np.nansum(k2_planet_masks[inds], axis=0) == 0) &
            (k2_lc['y']<= np.nanmedian(k2_lc['y'])+3.5*np.nanstd(k2_lc['y'])) )
    test = LC(time=k2_lc['x'][mask], 
              flux=k2_lc['y'][mask], 
              flux_err=k2_lc['yerr'][mask])


fig = plt.figure(constrained_layout=True, figsize=(28,14))
fig.set_facecolor('w')
gs_sq = fig.add_gridspec(2,4, hspace=0.3)#, height_ratios=[2,2,1.5])
gq_rt = fig.add_gridspec(2,4, hspace=0.5, wspace=0.3)

plt.rcParams['font.size'] = 18
bigger=26

fig.set_facecolor('w')
axes = [fig.add_subplot(gs_sq[0, 0]),
        fig.add_subplot(gs_sq[0, 1]), 
        fig.add_subplot(gs_sq[1, 0]), 
        fig.add_subplot(gs_sq[1, 1])]
last = fig.add_subplot(gq_rt[0, 2:])

ax1 = fig.add_subplot(gq_rt[1,2], projection=dss_wcs)
ax2 = fig.add_subplot(gq_rt[1,3], projection=dss_wcs)

alpha = 0.6

#ax1.figure.set_size_inches((8,6))
i =0
rt = [reproj_tesscut1, reproj_tesscut2]

for ax in [ax1,ax2]:
    # show image 1
    ax.imshow(dss_data, origin='lower', 
              cmap='Greys')

    # overlay image 2
    img = ax.imshow(rt[i], 
                    origin='lower', 
                    alpha=0.8, 
                    vmax=0.07, vmin=0.03, cmap='magma_r')

    ra,dec = 61.331630057213, 20.1571009815486

    ax.scatter(tic_ra[1:],#-0.0007, 
               tic_dec[1:],#+0.00015,
               marker='x', 
               transform=ax.get_transform('world'), s=150, #facecolors='none', 
               color='deepskyblue',
               linewidths=2)

    ax.scatter(ra, 
               dec,
               marker='o', 
               transform=ax.get_transform('world'), facecolors='none', 
               edgecolors='w', s=300,
               linewidths=2)
    ax.set_xlim(80,dss_data.shape[0])
    ax.set_ylim(50,dss_data.shape[0]-50)
    
    ax.set_xlabel('Right Ascension',fontsize=bigger)
    ax.set_ylabel('Declination',fontsize=bigger)
    
    xax = ax.coords[0]
    yax = ax.coords[1]
    xax.set_ticks(spacing=1*u.arcmin)
    yax.set_ticks(spacing=1*u.arcmin)
    
    if i == 0:
        ax.set_title('V1298 Tau b',y=1.05,fontsize=bigger)
    else:
        ax.set_ylabel('  ')
        ax.set_title('V1298 Tau e',y=1.05,fontsize=bigger)
    i += 1

cax = plt.axes([0.91, 0.14, 0.01, 0.28])
cbar = plt.colorbar(img, cax=cax)
cbar.set_label(r'$R_p/R_\star$', fontsize=bigger)
cbar.set_ticks(np.arange(0.02,0.08,0.01))

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
                     marker='o', color='w', 
                     markeredgecolor=k2_ec,
                     zorder=1, label='TESS data', 
                     ecolor=k2_ec, linestyle='')
    
    axes[i].plot(md.time.value,
                 md.flux.value, c=parula[i],
                 zorder=2, lw=lw)

    inds = np.delete(np.arange(0,4,1,dtype=int),i)
    mask = ((np.nansum(k2_planet_masks[inds], axis=0) == 0) &
            (k2_lc['y']<= np.nanmedian(k2_lc['y'])+3.5*np.nanstd(k2_lc['y'])) )
    
    test = LC(time=k2_lc['x'][mask], 
              flux=k2_lc['y'][mask], 
              flux_err=k2_lc['yerr'][mask])
    lkk2 = test.fold(t0=k2_t0[i], period=k2_pd[i])
    
    axes[i].errorbar(lkk2.time.value,
                     lkk2.flux.value,
                     yerr=lkk2.flux_err.value,
                     color=edgecolor, marker='o', 
                     zorder=3, label='K2 data', linestyle='')

    axes[i].set_xticks(np.arange(-1,1.5,0.5))
    rax.set_xticks(np.arange(-1,1.5,0.5))
    axes[i].set_xticklabels([])
    
    rax.errorbar(lk.time.value, 
                 lk.flux.value-md.flux.value, 
                 yerr=lk.flux_err.value,
                 marker='o', color='w', 
                    markeredgecolor=k2_ec,
                     zorder=1, label='TESS data', 
                     ecolor=k2_ec, linestyle='')
    
    #rax.plot(lk.time.value, 
    #         lk.flux.value-lkk2.flux.value, 'ko', zorder=1)
    
    rax.set_xticks(np.arange(-1,1.5,0.5))
    rax.set_ylim(-3,3)
    
    if i == 0:
        axes[i].legend(bbox_to_anchor=(0.4, 1.15, 0.7, .102), loc='lower left',
                       ncol=2,borderaxespad=0., fontsize=bigger)
        
    if i == 0 or i == 2:
        axes[i].set_ylabel('De-trended\nFlux [ppt]', labelpad=18, fontsize=bigger)
        rax.set_ylabel('Residuals', fontsize=bigger)
    if i ==2:
        rax.set_xlabel('Time from Mid-Transit [days]', fontsize=bigger,
                       x=1.15)
    
    axes[i].set_xlim(-0.2,0.2)
    rax.set_xlim(-0.5,0.5)
    axes[i].set_ylim(-7,3)
    axes[i].set_rasterized(True)

k2errs = [0.34, 0.43, 0.55, 0.78]
rpre = [5.59, 6.41, 10.27, 8.74]
lims = [5,10.5]
key = "('posterior', 'r_pl_rade[{}]', {})"

k2errs = [0.0017, 0.0022, 0.0023, 0.004]
rprs = [0.0381, 0.0436, 0.0700, 0.0611]
lims = [0.032, 0.075]
key = "('posterior', 'ror[{}]', {})"

for i, rp in enumerate(rprs):
    last.errorbar(rp, 
                  np.nanmedian(ror[i]),
                  xerr=k2errs[i],
                  yerr=np.nanstd(ror[i]),
                  marker='o', color=parula[i], ms=10)
last.plot(np.linspace(0,20,10), np.linspace(0,20,10), 'k')

last.set_xlim(lims)
last.set_ylim(lims)

gq_rt.update(left=0.2)

ax1.set_rasterized(True)
ax2.set_rasterized(True)

last.set_xlabel('K2 $R_p/R_{star}$', fontsize=bigger)
last.set_ylabel('TESS $R_p/R_{star}$', fontsize=bigger)
    
plt.subplots_adjust()
plt.savefig('compare_together.png', rasterize=True, 
            dpi=250,
            bbox_inches='tight')
