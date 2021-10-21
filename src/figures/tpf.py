import os
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
import matplotlib.pyplot as plt
from reproject import reproject_interp
import matplotlib.gridspec as gs
from matplotlib.colors import LogNorm
from astropy.visualization import simple_norm

COLOR = 'k'

plt.rcParams['font.size'] = 16
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

pixelsize = 11

# select a target and cutout size in pixels
ra, dec = 61.331654, 20.157032

target_x = 217
target_y = 1940

target = '{0}, {1}'.format(ra, dec)
x = y = pixelsize ## pixel size
# set local file path to current working directory
path = '../../data'
ffi = 'hlsp_tica_tess_ffi_s0043-o1-00180084-cam3-ccd2_tess_v01_img.fits'
sdss = 'frame-i-004334-6-0131.fits'

# grab the first file in the list
hdu = fits.open(os.path.join(path, ffi))

#tpf = k2flix.TargetPixelFile(filename)
n_pix = hdu[0].data.shape[0]
res = 21.0 * (u.arcsec/u.pixel) ## I think it's actually 20.98 or something. This is fine. 
area = res * (n_pix * u.pixel)
d = area.to(u.degree)
fov = d.value 

# compute the wcs of the image
wcs = WCS(hdu[0].header)#tpf.hdulist['APERTURE'].header)


## Grab nearby star data
# I'm worried this will cause issues if MAST is down, 
#   so right now I'm importing a saved table
#catalogData = Catalogs.query_region(target, radius=1, catalog="Tic")
#catalogData[np.where(catalogData['Tmag']<14)]
catalogData = Table.read(os.path.join(path, 'catalogData.tab'), format='ascii')

# get RA and Dec coords of catalog data
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


# compute the pixel area in arcmin
data = hdu[0].data + 0.0
norm = simple_norm(data, 'log')
arcmin = area.to(u.arcmin).value
# retrieve the DSS file
dss = fits.open(os.path.join(path, sdss))
dss_data = dss[0].data
dss_wcs = WCS(dss[0].header)

q = ((tic_ra<61.4) & (tic_ra>61.26) * (tic_dec>20.13) & (tic_dec<20.2))

reproj_tesscut, footprint = reproject_interp((data, wcs,), 
                                             dss_wcs, 
                                             shape_out=dss_data.shape, 
                                             order='nearest-neighbor')

plt.rcParams.update({'font.size': 15})
alpha = 0.6

fig = plt.figure(figsize=(10,10))
fig.set_facecolor('w')
ax = plt.subplot(projection=dss_wcs)
#ax.figure.set_size_inches((10,10))

# show image 1
ax.imshow(dss_data, origin='lower', cmap='Greys',
          vmax=2)

# overlay image 2
ax.imshow(dss_data, origin='lower', cmap='Greys',
          vmin=0, vmax=2)#, alpha=0.5)   

img = ax.imshow(reproj_tesscut, origin='lower', 
                alpha=0.5, 
                norm=LogNorm(vmin=1e5, vmax=1e8))

plt.colorbar(img, ax=ax,label=r'Flux (e$^{-1}$ s$^{-1}$)')

ax.scatter(tic_ra[q][1:],#-0.0007, 
           tic_dec[q][1:],#+0.00015,
           marker='x', 
           transform=ax.get_transform('world'), s=100, #facecolors='none', 
           color='darkorange',
           linewidths=2)
ax.scatter(tic_ra[q][0],#-0.0007, 
           tic_dec[q][0],#+0.00015,
           marker='o', 
           transform=ax.get_transform('world'), facecolors='none', 
           edgecolors='w', s=300,
           linewidths=2)

# add axis labels
ax.set_xlabel('Right Ascension',fontsize=20)
ax.set_ylabel('Declination',fontsize=20)
xlim = ax.get_xlim()
ylim = ax.get_ylim()

ax.set_xlim(0,750)
ax.set_ylim(0,800)
ax.set_rasterized(True)

plt.savefig('TESSaperture.pdf',dpi=300,
            bbox_inches='tight', 
            rasterize=True)
