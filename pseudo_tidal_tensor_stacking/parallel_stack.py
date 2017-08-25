import sys
import time
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
import matplotlib.ticker as ticker
from functools import partial

#################################################################################
# This script stacks HEALpix data in an oriented manner by centering on cluster #
# coordinates and rotating by the angle determined in num_density.py		#
#################################################################################

""" program setup """

num_cores = mp.cpu_count() / 2 	# take half of available cores for parallelization
orient 	  = False		# collects and saves oriented sections of map
combine   = True		# combines map sections and takes avg, med, sum
plots 	  = True		# plots stacked maps
size 	  = 4


""" defining correct filename that contains wanted positions and angles """

sep = 50			# separation in Mpc in search for neighbouring clusters
sset = "redmapper_mock_smoothed_strain_2e6-1e6"	# subset of haloes

name = "{}_pp_{}".format(sep, sset)


""" download HEALpix map if not already done """

try:
	peakpatch
except (NameError):
	#peakpatch = hp.read_map('/mnt/scratch-lustre/cbevingt/pp/tSZ_8Gpc_n4096_nb23_nt18_13579_nside2048_hp.fits')
	peakpatch = hp.read_map('/mnt/scratch-lustre/cbevingt/pp/2260Mpc_octant_hp.fits')
	pp = hp.sphtfunc.smoothing(peakpatch, fwhm=0.002)
	#pp = peakpatch

""" download cluster catalog if not already done """

try:
	peakdata
except (NameError):
	"""
	nhalos 	= 38916926
	ninfo  	= 5
	cat	= np.fromfile("/mnt/scratch-lustre/cbevingt/pp/tSZ_halos_zlt_1pt25.bin", \
			      dtype="float32", count = nhalos * ninfo)
	keep 	= (np.loadtxt('/mnt/scratch-lustre/cbevingt/vectors/{}/subset_{}.dat'.format(name,name)).astype(int))
	cat	= cat.reshape(38916926,5)

	x_pp    = cat[:,0] 
	y_pp    = cat[:,1] 
	z_pp    = cat[:,2] 
	vrad_pp = cat[:,3] 
	M_pp    = cat[:,4]
	
	z_cut = [350, 2000]

	chi	= np.sqrt(x_pp**2 + y_pp**2 + z_pp**2)
	gtr = (chi >= z_cut[0]).astype(int)
	lss = (chi <= z_cut[1]).astype(int)

	subset = ((gtr+lss)/2).astype(bool)

	x_vec	= x_pp[subset][keep] / chi[subset][keep]
	y_vec	= y_pp[subset][keep] / chi[subset][keep]
	z_vec	= z_pp[subset][keep] / chi[subset][keep]
	c 	= SkyCoord(x=x_pp[subset][keep], y=y_pp[subset][keep], z=z_pp[subset][keep], \
			   unit=u.Mpc, representation='cartesian')
	l, b	= c.transform_to('icrs').ra.value, c.transform_to('icrs').dec.value
	"""
	infile		= open('/mnt/scratch-lustre/cbevingt/pp/2260Mpc_n256_nb34_nt9_merge.pksc.13579', 'rb')
	Nhalo         	= np.fromfile(infile,dtype=np.int32,count=1)
	RTHmax 		= np.fromfile(infile,dtype=np.float32,count=1)
	zin           	= np.fromfile(infile,dtype=np.float32,count=1)
	keep 	= (np.loadtxt('/mnt/scratch-lustre/cbevingt/vectors/{}/subset_{}.dat'.format(name,name)).astype(int))
	print "\nNumber of halos to read in = ",Nhalo[0] 
	  
	outnum    	= 33 
	npkdata   	= outnum*Nhalo
	peakdata 	= np.fromfile(infile, dtype=np.float32, count=npkdata)
	peakdata 	= np.reshape(peakdata,(Nhalo[0],outnum)) 
	 
	x_pp   		= peakdata[:,0]
	y_pp	   	= peakdata[:,1]
	z_pp      	= peakdata[:,2]
	chi		= np.sqrt(x_pp**2 + y_pp**2 + z_pp**2)

	x_cond		= (x_pp>0).astype(int)
	y_cond		= (y_pp>0).astype(int)
	z_cond		= (z_pp>0).astype(int)
	chi_cond	= (chi <= cosmo.comoving_distance(0.6).value).astype(int)
	cone		= ((x_cond+y_cond+z_cond+chi_cond)/4).astype(bool)

	x_pp		= x_pp[cone]
	y_pp		= y_pp[cone]
	z_pp		= z_pp[cone]
	chi		= chi[cone]
	Rth  		= peakdata[:,6][cone]
	M_pp   		= 4./3 * np.pi * Rth**3 * 2.775e11 * 0.25 * 0.7**2
	strain 		= peakdata[:,14:20][cone]
	x_vec		= x_pp / chi
	y_vec		= y_pp / chi
	z_vec		= z_pp / chi
	
	e11	= strain[:,0]
	e22	= strain[:,1]
	e33	= strain[:,2]
	e23	= strain[:,3]
	e13	= strain[:,4]
	e12	= strain[:,5]

	c 	= SkyCoord(x=x_pp[keep], y=y_pp[keep], z=z_pp[keep], \
			   unit=u.Mpc, representation='cartesian')
	l, b	= c.transform_to('icrs').ra.value, c.transform_to('icrs').dec.value

""" stacking parameters """


v = np.loadtxt('/mnt/scratch-lustre/cbevingt/vectors/{}/ells_{}.dat'.format(name,name))
theta_unordered = v[:,0]					# rotation angles
order = v[:,1].astype(int)
theta = theta_unordered[order]
N = len(theta)				# total number of clusters to potentially stack
res = 160					# size of map section for each cluster in pixels (res x res)
index = np.arange(N)				# define an index for naming files for each oriented map section
TIC = time.clock()
#!!! REMOVE AFTER RAND !!!
#name += "_RAND"
#theta = 360*np.random.rand(N)
""" stacking function """



def stack(res, args):

	#########################################################################
	# Takes the coordinates and rotation angle for a given cluster, centres	#
	# a res x res gnomonic projection, rotates by the provided angle, and	#
	# saves the results to a file for later combination			#
	#########################################################################

	# theta=0 implies no neighbours within sep were found;
	# skip these instances as they add noise to stacked map
	lon, lat, theta, index, q = args
	if theta != 0:
		# calculate res x res gnomonic projection
		ymap = hp.gnomview(pp, rot=(lon, lat, theta), xsize=res, reso=size*60./res, return_projected_map=True)
		# healpy opens a figure for each map, close since we only want the grid
		plt.close("all")
		# save a flattened version of the grid to a file named with current cluster index
		np.savetxt('/mnt/scratch-lustre/cbevingt/map_out/{}/{}_{}_{}x{}'.format(name, name, index, size, size), \
			   ymap.data[::-1].flatten())
	q.put(index)

def listener(q):
	count = 0
	while 1:
		m = q.get()
		if m == 'kill':
			break
		else:
			count += 1
		sys.stdout.write('\r{:5.2f}% |{}{}|'.format(100.*(count+1)/N, '='*(50*(count+1)/N), ' '*(50-50*(count+1)/N)))
		sys.stdout.flush()



""" parallelize the stacking procedure """

if __name__ == '__main__' and orient:
	print "\nBeginning Stacking..."
	manager = mp.Manager()
	q = manager.Queue()
	pool = mp.Pool(num_cores)				# instantiate a worker pool
	chunksize, extra = divmod(N, 4*num_cores)		# dataset is large; send in chuncks to workers
	if extra:
		chunksize += 1
	func = partial(stack, res)				# res is known for all passes to stack

	watcher = pool.apply_async(listener, (q,))

	# pass cluster and orientation data to pool, wrapped in tqdm for progress updates
	for _ in (pool.imap_unordered(func, ((l[i], b[i], theta[i], index[i], q,) for i in range(N)), chunksize=chunksize)):
		pass
	q.put('kill')
	pool.close()						# don't allow anymore processes to be sent to pool
	pool.join()						# wait for worker processes to exit


""" combine individual map sections """

if combine:

	
	print "\nReading in data..."	
	#y = []
	y = np.zeros(res*res)
	for i in tqdm(range(N)):
		# recall if theta=0 or the cluster is in a masked region no map was created and hence no file
		# for that cluster index exists
		try:
			# try to upload flattened array, reshape to grid
			y0 = np.fromfile('/mnt/scratch-lustre/cbevingt/map_out/{}/{}_{}_{}x{}'.format(name, name, index[i], size, size), count=res*res, sep='\n')
			# append to combined array
			#y.append(y0)
			y += y0
		except (IOError):
			# if the file doesn't exist, move onto next cluster
			pass
	
	"""
	def read_in(i):
		try:
			y0 = np.fromfile('/mnt/scratch-lustre/cbevingt/map_out/{}/{}_{}_{}x{}'.format(name, name, index[i], size, size), count=res*res, sep='\n').reshape((res,res))
			sys.stdout.write('\rReading in stack: {:6.0f}'.format(index[i]+1))
			sys.stdout.flush()
			return y0
		except (IOError):
			pass

	if __name__ == '__main__':
		pool = mp.Pool(num_cores)
		y = pool.map(read_in, range(N))
			
	"""
	

	y = 1e6*y.reshape((res,res))	# convert to numpy array to easier manipulation
	#y_avg = np.mean(y, axis=0)	# average stacked map
	#y_med = np.median(y, axis=0)	# median stacked map
	#y_sum = np.sum(y, axis=0)	# summed stacked map
	y_sum = y
	y_avg = y/N

	# perhaps we're more interested in the log of stacked maps to accentuate features

	y_log_avg = np.zeros((res, res))
	for i in range(res):
		for j in range(res):
			# get rid of low and negative values
			if y_avg[i][j] > 0.01:
				y_log_avg[i][j] = y_avg[i][j]
			else:
				y_log_avg[i][j] = 0.01
	"""
	y_log_med = np.zeros((res, res))
	for i in range(res):
		for j in range(res):
			if y_med[i][j] > 0.01:
				y_log_med[i][j] = y_med[i][j]
			else:
				y_log_med[i][j] = 0.01
	"""
	y_log_sum = np.zeros((res, res))
	for i in range(res):
		for j in range(res):
			if y_sum[i][j] > 1:
				y_log_sum[i][j] = y_sum[i][j]
			else:
				y_log_sum[i][j] = 1
	

	# save these stacked maps to the disk
	np.savetxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_AVG_{}_{}x{}'.format(name, name, size, size), y_avg.flatten())
	#np.savetxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_MED_{}_{}x{}'.format(name, name, size, size), y_med.flatten())
	np.savetxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_SUM_{}_{}x{}'.format(name, name, size, size), y_sum.flatten())
	
	# plot results if necessary
if plots:
	print "\nBeginning plotting..."
	if not combine:
		y_avg = np.loadtxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_AVG_{}_{}x{}'.format(name, name, size, size)).reshape(res,res)
		#y_med = np.loadtxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_MED_{}_{}x{}'.format(name, name, size, size)).reshape(res,res)
		y_sum = np.loadtxt('/mnt/scratch-lustre/cbevingt/map_out/{}/Y_SUM_{}_{}x{}'.format(name, name, size, size)).reshape(res,res)
	
		y_log_avg = np.zeros((res, res))
		for i in range(res):
			for j in range(res):
				# get rid of low and negative values
				if y_avg[i][j] > 0.01:
					y_log_avg[i][j] = y_avg[i][j]
				else:
					y_log_avg[i][j] = 0.01
		"""
		y_log_med = np.zeros((res, res))
		for i in range(res):
			for j in range(res):
				if y_med[i][j] > 0.01:
					y_log_med[i][j] = y_med[i][j]
				else:
					y_log_med[i][j] = 0.01
		"""
		y_log_sum = np.zeros((res, res))
		for i in range(res):
			for j in range(res):
				if y_sum[i][j] > 1:
					y_log_sum[i][j] = y_sum[i][j]
				else:
					y_log_sum[i][j] = 1

	""" stacked maps """

	plt.figure()
	plt.imshow(y_log_avg, cmap="jet", extent=[-size/2,size/2,-size/2,size/2], norm=LogNorm())
	plt.title("Peak Patch Haloes: average stacked")
	plt.xlabel('[deg]')
	plt.ylabel('[deg]')
	plt.colorbar(ticks=[min(y_log_avg.flatten()), \
			    0.5*(max(y_log_avg.flatten()) - min(y_log_avg.flatten())) + min(y_log_avg.flatten()), \
			    max(y_log_avg.flatten())], format='%.2f')
	plt.draw()
	plt.savefig("LOG_AVG_STACKED_{}_{}x{}".format(name, size, size), dpi=500)
	"""
	plt.figure()
	plt.imshow(y_log_med, cmap="jet", extent=[-size/2,size/2,-size/2,size/2], norm=LogNorm())
	plt.title("Peak Patch Haloes: median stacked")
	plt.xlabel('[deg]')
	plt.ylabel('[deg]')
	plt.colorbar(ticks=[min(y_log_med.flatten()), \
			    0.5*(max(y_log_med.flatten()) - min(y_log_med.flatten())) + min(y_log_med.flatten()), \
			    max(y_log_med.flatten())], format='%.2f')
	plt.draw()
	plt.savefig("LOG_MED_STACKED_{}_{}x{}".format(name, size, size), dpi=500)
	"""	
	plt.figure()
	plt.imshow(y_log_sum, cmap="jet", extent=[-size/2,size/2,-size/2,size/2], norm=LogNorm())
	plt.title("Peak Patch Haloes: sum stacked")
	plt.xlabel('[deg]')
	plt.ylabel('[deg]')
	plt.colorbar(ticks=[min(y_log_sum.flatten()), \
			    0.5*(max(y_log_sum.flatten()) - min(y_log_sum.flatten())) + min(y_log_sum.flatten()), \
			    max(y_log_sum.flatten())], format='%.2e')
	plt.draw()
	plt.savefig("SUM_STACKED_{}_{}x{}".format(name, size, size), dpi=500)
	plt.show()
