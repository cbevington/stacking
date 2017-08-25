import sys
import multiprocessing as mp
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
import healpy as hp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from functools import partial

#################################################################################
# This script stacks HEALpix data in an oriented manner by centering on cluster #
# coordinates and rotating by the angle determined in `cat_angles.py`		#
#################################################################################

""" program setup """

num_cores = mp.cpu_count() / 2 	# specify number of cores for parallelization
spec      = False		# clusters have only spectroscopic redshifts if True
orient 	  = False 		# collects and saves oriented sections of map
combine   = False		# combines map sections and takes avg, med, sum
plots     = True		# plots stacked maps
sep       = 50			# separation in Mpc in search for neighbouring clusters
name 	  = "50_rand"		# base filename
catname   = "rand_cls.fits"		# catalog filename


""" download HEALpix map if not already done """

try:
	planck
except (NameError):
	planck 	    	= hp.ma(hp.read_map('/home/cbevingt/proj/nilc_ymaps.fits'))
	mask 		= hp.read_map('/home/cbevingt/proj/masks.fits')
	planck.mask 	= mask == 0
	

""" download cluster catalog if not already done """

try:		
	cat
except (NameError):
	hdu 	= fits.open('{}'.format(catname))
	cat 	= hdu[1].data
	z_sp 	= cat.field('zph')
	#z_sp 	= cat.field('zsp')

if spec:
	pass
#	sp_vals	= np.logical_not(np.isnan(z_sp))	# remove clusters w/ non-spectroscopic z
#	z 	= zsp[sp_vals]
#	ra 	= cat.field('RAJ2000')[sp_vals]
#	dec 	= cat.field('DEJ2000')[sp_vals]
	#rich 	= cat.field('RL*')[sp_vals]
else:						
	z 	= z_sp[:]				# otherwise keep all clusters
	for i in range(len(z)):
		if np.isnan(z_sp[i]):	# locate non-spectroscopic
			z[i] = z_ph[i]	# replace with estimated z
	ra 	= cat.field('RAJ2000')
	dec 	= cat.field('DEJ2000')
	#rich 	= cat.field('RL*')

#keep	= np.loadtxt('/mnt/scratch-lustre/cbevingt/vectors/{}/subset_{}.dat'.format(name,name)).astype(int) 
c  	= SkyCoord(ra*u.deg, dec*u.deg, cosmo.comoving_distance(z))	# intialized coordinates
#c 	= c[keep].galactic
c	= c.galactic						# convert to galactic
l  	= c.l.value
b  	= c.b.value


""" stacking parameters """

v 	= np.loadtxt('/mnt/scratch-lustre/cbevingt/vectors/{}/ells_{}.dat'.format(name,name))
theta 	= v[:,1]
N 	= len(theta)				# total number of clusters to potentially stack
res 	= 160					# size of map section for each cluster in pixels (res x res)
index 	= np.arange(N)				# define an index for naming files for each oriented map section


""" stacking function """

def stack(res, args):

	#########################################################################
	# Takes the coordinates and rotation angle for a given cluster, centres	#
	# a res x res gnomonic projection, rotates by the provided angle, and	#
	# saves the results to a file for later combination			#
	#########################################################################

	lon, lat, theta, index = args
	# theta=0 implies no neighbours within sep were found;
	# skip these instances as they add noise to stacked map
	if theta != 0:
		# calculate res x res gnomonic projection
		ymap = hp.gnomview(planck, rot=(lon, lat, theta), xsize=res, reso=1.5, return_projected_map=True)
		# healpy opens a figure for each map, close since we only want the grid
		plt.close("all")
		# save a flattened version of the grid to a file named with current cluster index
		if not ymap.mask.any():
			np.save('/mnt/scratch-lustre/cbevingt/map_out/{}/{}_{}'.format(name, name, index), \
			   	ymap.data[::-1])


""" parallelize the stacking procedure """

if __name__ == '__main__' and orient:
	pool = mp.Pool(num_cores)				# instantiate a worker pool
	chunksize, extra = divmod(N, 4*num_cores)		# dataset is large; send in chuncks to workers
	if extra:
		chunksize += 1
	func = partial(stack, res)				# res is known for all passes to stack

	# pass cluster and orientation data to pool, wrapped in tqdm for progress updates
	for _ in tqdm(pool.imap_unordered(func, ((l[i], b[i], theta[i], index[i],) for i in range(N)), chunksize=chunksize)):
		pass
	pool.close()						# don't allow anymore processes to be sent to pool
	pool.join()						# wait for worker processes to exit


""" combine individual map sections """

def read_in(i):
	try:
		y0 = np.load('/mnt/scratch-lustre/cbevingt/map_out/{}/{}_{}.npy'.format(name, name, index[i]))
		if 50*np.random.rand() > 49 or i == N-1:
			sys.stdout.write('\r[{}{}] {:0.1f}%'.format('='*(50*(i+1)/N), ' '*(50-50*(i+1)/N), 100.*(i+1)/N))
			sys.stdout.flush()
		return y0
	except (IOError):
		pass

	

if combine and __name__ == '__main__':

	pool = mp.Pool(num_cores)
	y = pool.map(read_in, range(N))
	y = [i for i in y if i is not None]
	y = [i for i in y if not (np.absolute(i) > 1e10).any()]
	y = 1e6*np.array(y)			# convert to numpy array to easier manipulation	
	y_avg = np.mean(y, axis=0)	# average stacked map
	y_med = np.median(y, axis=0)	# median stacked map
	y_sum = np.sum(y, axis=0)	# summed stacked map

	# perhaps we're more interested in the log of stacked maps to accentuate features

	y_log_avg = np.zeros((res, res))
	for i in range(res):
		for j in range(res):
			# get rid of low and negative values
			if y_avg[i][j] > 0.01:
				y_log_avg[i][j] = y_avg[i][j]
			else:
				y_log_avg[i][j] = 0.01

	y_log_med = np.zeros((res, res))
	for i in range(res):
		for j in range(res):
			if y_med[i][j] > 0.00001:
				y_log_med[i][j] = y_med[i][j]
			else:
				y_log_med[i][j] = 0.00001

	y_log_sum = np.zeros((res, res))
	for i in range(res):
		for j in range(res):
			if y_sum[i][j] > 1:
				y_log_sum[i][j] = y_sum[i][j]
			else:
				y_log_sum[i][j] = 1
	

	# save these stacked maps to the disk
	np.savetxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_AVG_{}'.format(name, name), y_avg.flatten())
	np.savetxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_MED_{}'.format(name, name), y_med.flatten())
	np.savetxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_SUM_{}'.format(name, name), y_sum.flatten())
	
	# plot results if necessary
if plots:
	if not combine:
		y_avg = np.loadtxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_AVG_{}'.format(name, name)).reshape(res,res)
		y_med = np.loadtxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_MED_{}'.format(name, name)).reshape(res,res)
		y_sum = np.loadtxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_SUM_{}'.format(name, name)).reshape(res,res)
	
		y_log_avg = np.zeros((res, res))
		for i in range(res):
			for j in range(res):
				# get rid of low and negative values
				if y_avg[i][j] > 0.01:
					y_log_avg[i][j] = y_avg[i][j]
				else:
					y_log_avg[i][j] = 0.01

		y_log_med = np.zeros((res, res))
		for i in range(res):
			for j in range(res):
				if y_med[i][j] > 0.00001:
					y_log_med[i][j] = y_med[i][j]
				else:
					y_log_med[i][j] = 0.00001

		y_log_sum = np.zeros((res, res))
		for i in range(res):
			for j in range(res):
				if y_sum[i][j] > 1:
					y_log_sum[i][j] = y_sum[i][j]
				else:
					y_log_sum[i][j] = 1

	""" stacked maps """

	plt.figure()
	plt.imshow(y_log_avg, cmap="jet", extent=[-2,2,-2,2], norm=LogNorm())
	plt.title("Random positions: average stacked")
	plt.xlabel('[deg]')
	plt.ylabel('[deg]')
	plt.colorbar()
	plt.draw()
	plt.savefig("LOG_AVG_STACKED_{}".format(name), dpi=500)

	plt.figure()
	plt.imshow(y_log_med, cmap="jet", extent=[-2,2,-2,2], norm=LogNorm())
	plt.title("Random positions: median stacked")
	plt.xlabel('[deg]')
	plt.ylabel('[deg]')
	plt.colorbar()
	plt.draw()
	plt.savefig("LOG_MED_STACKED_{}".format(name), dpi=500)
	
	plt.figure()
	plt.imshow(y_log_sum, cmap="jet", extent=[-2,2,-2,2], norm=LogNorm())
	plt.title("Random positions: sum stacked")
	plt.xlabel('[deg]')
	plt.ylabel('[deg]')
	plt.colorbar()
	plt.draw()
	plt.savefig("SUM_STACKED_{}".format(name), dpi=500)
	plt.show()
