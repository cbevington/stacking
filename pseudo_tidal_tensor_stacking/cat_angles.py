import multiprocessing as mp
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
import scipy.interpolate as interp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from matplotlib.patches import Ellipse

#########################################################################################
# This script calculates the rotation angle for oriented stacking, given a catalog of	#
# clusters. The rotation angle is determined by a mock tidal tensor based on		#
# neighbouring clusters in 3D. These angles are saved to the disk for later oriented 	#
# stacking, completed in `cat_stack.py`.						#
#########################################################################################



##################################################################################################################
""" program setup """
##################################################################################################################

num_cores = mp.cpu_count() / 2 	# specify number of cores for parallelization
spec      = False		# clusters have only spectroscopic redshifts if True
sep       = 50			# separation in Mpc in search for neighbouring clusters
name      = "50_rand"		# base filename of output data
catname   = "asu.fit"		# catalog filename
c_cut     = False		# apply a coordinate cut
m_cut     = False		# apply a mass cut



##################################################################################################################
""" import catalog if not already done """
##################################################################################################################

try:		
	cat
except (NameError):
	hdu 	= fits.open('/home/cbevingt/proj/{}'.format(catname))
	cat 	= hdu[1].data
	z_ph 	= cat.field('zph')
	z_sp 	= cat.field('zsp')

if spec:
	sp_vals	= np.logical_not(np.isnan(z_sp))	# remove clusters w/ non-spectroscopic z
	z 	= zsp[sp_vals]
	ra 	= cat.field('RAJ2000')[sp_vals]
	dec 	= cat.field('DEJ2000')[sp_vals]
	#rich 	= cat.field('RL*')[sp_vals]
else:						
	z 	= z_sp[:]				# otherwise keep all clusters
	for i in range(len(z)):
		if np.isnan(z_sp[i]):	# locate non-spectroscopic
			z[i] = z_ph[i]	# replace with estimated z
	ra 	= cat.field('RAJ2000')
	dec 	= cat.field('DEJ2000')
	#rich 	= cat.field('RL*')

c = SkyCoord(ra*u.deg, dec*u.deg, cosmo.comoving_distance(z))	# intialized coordinates
c = c.galactic							# convert to galactic
l = c.l.value
b = c.b.value



##################################################################################################################
""" functions """
##################################################################################################################

def equal_edge(z, nbins):

	#################################################################################
	# This function divides all clusters into equally-populated redshift bins. Used	#
	# if you wish to split the stacking into multiple redshift populations or view	#
	# plots of the tidal vectors for redshift slices. Given an array of redshifts z	#
	# and a number of bins nbins, the function returns the corresponding bin edges.	#
	#################################################################################

	npts = len(z)
	return np.interp(np.linspace(0, npts, nbins+1), np.arange(npts), np.sort(z))


def norm_coord(l, b, l0, b0):

	#################################################################################
	# This function converts angular coordinates into gnomonic-projected "normal	#
	# coordinates". Used if proj = "gnom". Here (l0, b0) are the angular coords	#
	# of the central cluster of the projection, and (l, b) are the angular coords	#
	# of the nearby cluster needed to be projected.					#
	#################################################################################

	# start by converting from deg to rad
	l, b, l0, b0 = np.pi*np.array([l,b,l0,b0])/180

	# define denominator of the projection conversion
	cosc = np.sin(b)*np.sin(b0) + np.cos(b)*np.cos(b0)*np.cos(l0-l)

	# calculate (x, y) given (lon, lat)	
	x = np.cos(b0)*np.sin(l0-l)/cosc
	y = (np.cos(b)*np.sin(b0) - np.sin(b)*np.cos(b0)*np.cos(l0-l))/cosc
	return x,y


def pvector(i, q):
	c0 = coords[i]
	x0, y0, d0 = c0.l.value, c0.b.value, c0.distance.value			
	ang_max = 180. * (sep / d0) / np.pi
	in_z = (np.absolute(c.distance.value - d0) <= 100.).astype(int)
	in_rad = (c.separation(c0).value <= ang_max).astype(int)
	vals = ((in_z + in_rad)/2).astype(bool)

	x, y = norm_coord(c[vals].l.value, c[vals].b.value, x0, y0)
	rz = np.absolute(c[vals].distance.value - d0) * ang_max / sep + 0.01*np.random.rand()
	nr = x**2 + y**2
	vx, vy = np.sum(x[nr!=0]/(nr[nr!=0] * rz[nr!=0])), \
		 np.sum(y[nr!=0]/(nr[nr!=0] *  rz[nr!=0]))
	res = np.sqrt(vx**2 + vy**2), 180. * np.arctan2(vy,vx) / np.pi
	q.put(res)
	return res
	 


def listener(q):
	f = open('/mnt/scratch-lustre/cbevingt/vectors/{}/ells_{}.dat'.format(name, name), 'wb')
	while 1:
		m = q.get()
		if m == 'kill':
			break
		f.write(' '.join(str(i) for i in m) + '\n')
		f.flush()
	f.close()


def main():
	manager = mp.Manager()
	q = manager.Queue()
	pool = mp.Pool(num_cores)
	
	watcher = pool.apply_async(listener, (q,))
	
	jobs = []
	for i in range(N):
		job = pool.apply_async(pvector, (i, q))
		jobs.append(job)

	for job in tqdm(jobs):
		job.get()

	q.put('kill')
	pool.close()




##################################################################################################################
""" final preparations before calclating vectors """
##################################################################################################################

if c_cut:
	keep = []
	for i in range(len(c)):
		if ra[i] > 100 and ra[i] < 300:
			if (ra[i]-185)**2/66**2+(dec[i]-32.5)**2/33**2 <= 1:
				keep.append(i)
		else:
			if (dec[i] >= -5 and dec[i] <= 28) and \
			((ra[i] >=0 and ra[i] <= 25) or (ra[i] >= 335 and ra[i] <= 360)):
		     		keep.append(i)
	coords = c[keep]
else:
	rand_c 	= fits.open('rand_cls.fits')[1].data
	rra 	= rand_c.field('RAJ2000')
	rdec	= rand_c.field('DEJ2000')
	rzph	= rand_c.field('zph')
	coords 	= SkyCoord(rra*u.deg, rdec*u.deg, cosmo.comoving_distance(rzph))
	coords 	= coords.galactic
	

#np.savetxt('/mnt/scratch-lustre/cbevingt/vectors/{}/subset_{}.dat'.format(name, name), keep)
N = 50000 #N = len(keep)
index = np.arange(N)



##################################################################################################################
""" main program """
##################################################################################################################

if __name__ == "__main__":
	main()

