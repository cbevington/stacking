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
# clusters (real or simulated). The rotation angle is determined by a mock tidal force	#
# vector based on neighbouring clusters in 3D. These angles are saved to the disk for 	#
# later oriented stacking, completed in `parallel.py`.					#
#########################################################################################



##################################################################################################################
""" program setup """
##################################################################################################################

num_cores = mp.cpu_count() - 2 	# take half of available cores for parallelization
parallel = True			# run in parallel
ppatch = True			# using redMaPPer if False, ppatch if True
plots = False			# plot vector visualization
spec = False			# clusters have only spectroscopic redshifts if True
sep = 50			# separation in Mpc in search for neighbouring clusters
bin_num = "redmapper_mock_smoothed_strain_gaussmass1e6"	# "all" indicates all redshift bins will be combined
c_cut = False			# apply a coordinate cut
m_cut = True			# apply a mass cut
proj = "gnom"			# vectors are defined in gnomonic space if "gnon"
gal = True			# working in galactic coordinates if True



##################################################################################################################
""" define filename for later storage """
##################################################################################################################

if not ppatch and spec:
	name = "{}_z_spec_{}_{}".format(sep, bin_num, proj)
elif not ppatch:
	name = "{}_z_mixed_{}_{}".format(sep, bin_num, proj)
if not ppatch and gal:
	name += "_GAL"
if ppatch:
	name = "{}_pp_{}".format(sep, bin_num)



##################################################################################################################
""" import catalog if not already done """
##################################################################################################################

try:		
	cat
except (NameError):
	if ppatch:
		"""
		nhalos 	= 38916926
		ninfo  	= 5
		cat	= np.fromfile("/mnt/scratch-lustre/cbevingt/pp/tSZ_halos_zlt_1pt25.bin", \
				      dtype="float32", count = nhalos * ninfo)
		cat	= cat.reshape(38916926,5) 
		 
		x_pp    = cat[:,0] 
		y_pp    = cat[:,1] 
		z_pp    = cat[:,2] 
		vrad_pp = cat[:,3] 
		M_pp    = cat[:,4]

		chi	= np.sqrt(x_pp**2 + y_pp**2 + z_pp**2)
		x_vec	= x_pp / chi
		y_vec	= y_pp / chi
		z_vec	= z_pp / chi
		"""
		infile		= open('/mnt/scratch-lustre/cbevingt/pp/2260Mpc_n256_nb34_nt9_merge.pksc.13579', 'rb')
		Nhalo         	= np.fromfile(infile,dtype=np.int32,count=1)
		RTHmax 		= np.fromfile(infile,dtype=np.float32,count=1)
		zin           	= np.fromfile(infile,dtype=np.float32,count=1)
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
		n_x		= x_pp / chi
		n_y		= y_pp / chi
		n_z		= z_pp / chi
		
		e11	= strain[:,0]
		e22	= strain[:,1]
		e33	= strain[:,2]
		e23	= strain[:,3]
		e13	= strain[:,4]
		e12	= strain[:,5]

		s_tensor = np.array([[e11, e12, e13], [e12, e22, e23], [e13, e23, e33]])

	else:
		hdu 		= fits.open('redmapper_dr8_public_v6.3_catalog.fits')
		cat 		= hdu[1].data
		z_lambda 	= cat.field('Z_LAMBDA')
		z_spec 		= cat.field('Z_SPEC')

if not ppatch and spec:					# remove clusters w/ non-spectroscopic z
	z 	= z_spec[z_spec!=-1]
	ra 	= cat.field('RA')[z_spec!=-1]
	dec 	= cat.field('DEC')[z_spec!=-1]
	rich 	= cat.field('LAMBDA')[z_spec!=-1]
elif not ppatch:						
	z 	= z_spec[:]				# otherwise keep all clusters
	for i in range(len(z)):
		if z[i] == -1:			# non-spectroscopic z set to -1 in redMaPPer
			z[i] = z_lambda[i]	# replace with estimated z
	ra 	= cat.field('RA')
	dec 	= cat.field('DEC')
	rich 	= cat.field('LAMBDA')

if ppatch:
	c = SkyCoord(x=x_pp, y=y_pp, z=z_pp, unit=u.Mpc, representation='cartesian')
else:
	c = SkyCoord(ra, dec, unit='deg')		# intialized coordinates
	gc = c.galactic					# convert to galactic
	l = gc.l.value
	b = gc.b.value



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


def vectors(ra, dec, z, sep, k=False, z_cut=None, coord_cut=None, gal=True):

	#################################################################################
	# This function calculates the tidal vectors for each cluster given a set of 	#
	# coordinates, (ra, dec, z). 							#
	#										#
	# - sep is the anglar separation searched for neighbouring clusters.		#
	# - If k=True then the number of neighbours is counted and saved to disk.	#
	# - z_cut and coord_cut specify the redshift and coordinate ranges		#
	#   respectively of the clusters for which the vectors are determined 		#
	#   (note that neither z_cut nor coord_cut limit the set of clusters of 	#
	#   potential neighbouring clusters, to avoid edge effects).			#
	# - gal=True performs analysis in galactic coordinates.				#
	#										#
	# Returns at least vector information and the indices of clusters kept for 	#
	# analysis, plus possibly neighbour counts.					#
	#################################################################################

	# start by initializing coordinates
	coords = SkyCoord(ra*u.deg, dec*u.deg, cosmo.comoving_distance(z))
	N = len(coords)

	# apply the redshift cut if necessary
	if z_cut:
		gtr = (z >= z_cut[0]).astype(int)
		lss = (z <= z_cut[1]).astype(int)
		subset = ((gtr+lss)/2).astype(bool)
	# otherwise keep all values
	else:
		subset = np.array(len(coords)*[1]).astype(bool)
	
	# apply the coordinate cut if necessary. NOTE: currently setup for redMaPPer
	if coord_cut:
		keep = []
		for i in range(N):
			if subset[i]:
				if ra[i] > 100 and ra[i] < 300:
					if (ra[i]-185)**2/66**2+(dec[i]-32.5)**2/33**2 <= 1:
						keep.append(i)
				else:
					if (dec[i] >= -5 and dec[i] <= 28) and \
					((ra[i] >=0 and ra[i] <= 25) or (ra[i] >= 335 and ra[i] <= 360)):
				     		keep.append(i)
		coords_cut = coords[keep]
	# otherwise keep all values
	else:
		keep = np.arange(len(coords_cut))
		coords_cut = coords[keep]

	# define some empty arrays to store future vector data	
	n = len(coords_cut)
	r, theta = n*[0], n*[0]

	# if neighbour counts are requested, initialize an array
	if k:
		nghbrs = n*[0]

	# calculate vectors
	for i in tqdm(range(n)):
		if gal:
			# convert current cluster to galactic coords
			c0 = coords_cut.galactic[i]
			
			# unpack coordinates and comoving distance
			x0, y0, d0 = c0.l.value, c0.b.value, c0.distance.value
			
			# define angular range for neighbouring cluster searches			
			ang_max = 180. * (sep / d0) / np.pi

			# find the subset of clusters within 100 Mpc in redshift separation, and within ang_max
			in_z = (np.absolute(coords.distance.value - d0) <= 100.).astype(int)
			in_rad = (coords.separation(c0).value <= ang_max).astype(int)
			vals = ((in_z + in_rad)/2).astype(bool)

			# count neighbours if requested
			if k:
				nghbrs[i] = len(coords[vals]) - 1

			# convert cluster subset coordinates to normal coordinates
			x, y = norm_coord(coords.galactic[vals].l.value, coords.galactic[vals].b.value, x0, y0)
			
			# scale the comoving distance separation to angular units
			rz = np.absolute(coords[vals].distance.value - d0) * ang_max / sep

			# calculate weighting factors and vector components in coordinate space
			nr = x**2 + y**2
			vx, vy = np.sum(x[nr!=0]/(nr[nr!=0] * rz[nr!=0])), \
				 np.sum(y[nr!=0]/(nr[nr!=0] *  rz[nr!=0]))

			# return polar vector coordinates
			r[i] = np.sqrt(vx**2 + vy**2)
			theta[i] = 180. * np.arctan2(vy,vx) / np.pi
		
		# otherwise the same procedure but using (ra, dec)
		else:
			c0 = coords_cut[i]
			x0, y0, d0 = c0.ra.value, c0.dec.value, c0.distance.value
			ang_max = 180. * (sep / d0) / np.pi
			in_z = (np.absolute(coords.distance.value - d0) <= 100.).astype(int)
			in_rad = (coords.separation(c0).value <= ang_max).astype(int)
			vals = ((in_z + in_rad)/2).astype(bool)
			if k:
				nghbrs[i] = len(coords[vals]) - 1
			x, y = norm_coord(coords[vals].ra.value, coords[vals].dec.value, x0, y0)
			rz = np.absolute(coords[vals].distance.value - d0) * ang_max / sep
			nr = x**2 + y**2
			vx, vy = np.sum(x[nr!=0]/(nr[nr!=0] * rz[nr!=0])), \
				 np.sum(y[nr!=0]/(nr[nr!=0] *  rz[nr!=0]))
			r[i] = np.sqrt(vx**2 + vy**2)
			theta[i] = 180. * np.arctan2(vy,vx) / np.pi

	# return the appropriate results
	if k:
		return np.array(r), np.array(theta), np.array(nghbrs), np.array(keep)
	else:
		return np.array(r), np.array(theta), np.array(keep)

def pvector(i, q):
	c0 = coords_cut.galactic[i]
	x0, y0, d0 = c0.l.value, c0.b.value, c0.distance.value			
	ang_max = 180. * (sep / d0) / np.pi
	in_z = (np.absolute(coords.distance.value - d0) <= 100.).astype(int)
	in_rad = (coords.separation(c0).value <= ang_max).astype(int)
	vals = ((in_z + in_rad)/2).astype(bool)

	x, y = norm_coord(coords.galactic[vals].l.value, coords.galactic[vals].b.value, x0, y0)
	rz = np.absolute(coords[vals].distance.value - d0) * ang_max / sep
	nr = x**2 + y**2
	vx, vy = np.sum(x[nr!=0]/(nr[nr!=0] * rz[nr!=0])), \
		 np.sum(y[nr!=0]/(nr[nr!=0] *  rz[nr!=0]))
	res = np.sqrt(vx**2 + vy**2), 180. * np.arctan2(vy,vx) / np.pi
	q.put(res)
	return res

def pvector_pp(i, q):
	"""
	c0 = coords_cut[i]
	cc0 = c0.cartesian
	xc, yc, zc = cc0.x.value, cc0.y.value, cc0.z.value
	x0, y0, d0 = c0.ra.value, c0.dec.value, c0.distance.value			
	ang_max = 180. * (sep / d0) / np.pi
	in_z = (np.absolute(coords.distance.value - d0) <= sep).astype(int)
	in_rad = (coords.separation(c0).value <= ang_max).astype(int)
	vals = ((in_z + in_rad)/2).astype(bool)
	
	ccoords = coords[vals].cartesian
	x1, y1, z1 = ccoords.x.value, ccoords.y.value, ccoords.z.value
	dx, dy, dz = (x1-xc), (y1-yc), (z1-zc)

	x, y = norm_coord(coords[vals].ra.value, coords[vals].dec.value, x0, y0)
	#rz = np.absolute(coords[vals].distance.value - d0) * ang_max / sep
	nr = (dx**2 + dy**2 + dz**2)**1.5
	M = mass_cut[vals]
	vx, vy = np.sum(M[np.abs(nr)>1e-10]*x[np.abs(nr)>1e-10]/nr[np.abs(nr)>1e-10]), \
		 np.sum(M[np.abs(nr)>1e-10]*y[np.abs(nr)>1e-10]/nr[np.abs(nr)>1e-10])
	res = np.sqrt(vx**2 + vy**2), 180. * np.arctan2(vy,vx) / np.pi
	q.put(res)
	return res
	"""
	c0 = coords_cut[i]
	ra, dec = c0.ra.value, c0.dec.value
	r = hp.rotator.Rotator([ra, dec, 0])
	sT = np.matmul(r.mat, np.matmul(s_tensor_cut[:,:,i], r.mat.transpose()))
	evals, evecs = np.linalg.eigh(sT[1:,1:])
	evecA, evecB = evecs[:,0], evecs[:,1]
	if evecB[0] < 0:
		evecB = -evecB
	theta = np.arctan2(evecB[1], evecB[0])
	res = 180*theta.item()/np.pi, i
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
	#args = np.argsort(order)
	#np.savetxt('/mnt/scratch-lustre/cbevingt/vectors/{}/ells_{}.dat'.format(name, name), np.array(ells)[args])

def main():
	manager = mp.Manager()
	q = manager.Queue()
	pool = mp.Pool(num_cores)
	
	watcher = pool.apply_async(listener, (q,))
	
	jobs = []
	for i in range(N):
		if ppatch:
			job = pool.apply_async(pvector_pp, (i, q))
		else:
			job = pool.apply_async(pvector, (i, q))
		jobs.append(job)

	for job in tqdm(jobs):
		job.get()

	q.put('kill')
	pool.close()




##################################################################################################################
""" final preparations before calclating vectors """
##################################################################################################################

# create redshift bins if necessary
if bin_num == "all":
	z_cut = None
elif not ppatch:
	z_bins = equal_edge(z, 10)
	z_cut = [z_bins[bin_num-1], z_bins[bin_num]]
elif ppatch:
	z_cut = None #[350, 2000]

# if running in parallel, perform the cuts on the dataset done in the vector function beforehand
if not ppatch and parallel:

	coords = SkyCoord(ra*u.deg, dec*u.deg, cosmo.comoving_distance(z))

	if z_cut:
		gtr = (z >= z_cut[0]).astype(int)
		lss = (z <= z_cut[1]).astype(int)
		subset = ((gtr+lss)/2).astype(bool)
	else:
		subset = np.array(len(coords)*[1]).astype(bool)

	if c_cut:
		keep = []
		for i in range(len(subset)):
			if subset[i]:
				if ra[i] > 100 and ra[i] < 300:
					if (ra[i]-185)**2/66**2+(dec[i]-32.5)**2/33**2 <= 1:
						keep.append(i)
				else:
					if (dec[i] >= -5 and dec[i] <= 28) and \
					((ra[i] >=0 and ra[i] <= 25) or (ra[i] >= 335 and ra[i] <= 360)):
				     		keep.append(i)
		coords_cut = coords[keep]
	else:
		keep = np.arange(len(coords_cut))
		coords_cut = coords[keep]
	
	np.savetxt('/mnt/scratch-lustre/cbevingt/vectors/{}/subset_{}.dat'.format(name, name), keep)
	N = len(keep)
	index = np.arange(N)

if ppatch and parallel:

	coords = c
	keep = []

	if z_cut:
		gtr = (chi >= z_cut[0]).astype(int)
		lss = (chi <= z_cut[1]).astype(int)
		c_gtr = (chi >= z_cut[0] + sep).astype(int)
		c_lss = (chi <= z_cut[1] - sep).astype(int)
		subset = ((gtr+lss)/2).astype(bool)
		c_subset = ((c_gtr+c_lss)/2).astype(bool)
		coords = c[c_subset].transform_to('icrs')
	else:
		subset = np.array(len(coords)*[1]).astype(bool)
		c_subset = subset
		coords = c[c_subset].transform_to('icrs')

	if c_cut:
		for i in range(len(subset)):
			if subset[i]:
				if ra[i] > 100 and ra[i] < 300:
					if (ra[i]-185)**2/66**2+(dec[i]-32.5)**2/33**2 <= 1:
						keep.append(i)
				else:
					if (dec[i] >= -5 and dec[i] <= 28) and \
					((ra[i] >=0 and ra[i] <= 25) or (ra[i] >= 335 and ra[i] <= 360)):
				     		keep.append(i)
	if m_cut:
		keep = np.arange(1000000)				#1
		#keep = np.argsort(M_pp[subset])[-2000000:-1000000]	#2
		#keep = np.argsort(M_pp[subset])[:300000] 		#3
	else:
		keep = np.arange(len(coords_cut))

	coords_keep = np.argsort(M_pp[c_subset][:])
	coords = coords[coords_keep]	
	coords_cut = c[subset][keep].transform_to('icrs')
	s_tensor_cut = s_tensor[:,:,keep]
	mass_cut = M_pp[c_subset][coords_keep] / 1e13
	x_cs, y_cs, z_cs = x_pp[c_subset], x_pp[c_subset], x_pp[c_subset]
	x_k, y_k, z_k = x_pp[keep], y_pp[keep], z_pp[keep]
	
	np.savetxt('/mnt/scratch-lustre/cbevingt/vectors/{}/subset_{}.dat'.format(name, name), keep)
	N = len(keep)
	index = np.arange(N)



##################################################################################################################
""" main program """
##################################################################################################################

if parallel and __name__ == "__main__":
	main()

else:		
	try:
		v = np.loadtxt('ells_{}.dat'.format(name))
		r, theta = v[:,0], v[:,1]
		nghbrs = np.loadtxt('nghbrs_{}.dat'.format(name))
		vals = np.loadtxt('subset_{}.dat'.format(name)).astype(bool)
	except (IOError):
		try:
			nghbrs = np.loadtxt('nghbrs_{}.dat'.format(name))
			r, theta, vals = vectors(ra, dec, z, sep, z_cut=z_cut, coord_cut=c_cut, gal=gal)
			np.savetxt('/mnt/scratch-lustre/cbevingt/vectors/{}/ells_{}.dat'.format(name, name), \
				   np.array([[r[i], theta[i]] for i in range(len(r))]))
			np.savetxt('/mnt/scratch-lustre/cbevingt/vectors/{}/subset_{}.dat'.format(name, name), vals)
		except (IOError):
			r, theta, nghbrs, vals = vectors(ra, dec, z, sep, k=True, z_cut=z_cut, coord_cut=c_cut, gal=gal)
			np.savetxt('/mnt/scratch-lustre/cbevingt/vectors/{}/nghbrs_{}.dat'.format(name, name), nghbrs)
			np.savetxt('/mnt/scratch-lustre/cbevingt/vectors/{}/ells_{}.dat'.format(name, name), \
				   np.array([[r[i], theta[i]] for i in range(len(r))]))
			np.savetxt('/mnt/scratch-lustre/cbevingt/vectors/{}/subset_{}.dat'.format(name, name), vals)


""" plotting """

if plots:
	pre = np.zeros(len(theta))
	pre[np.nonzero(nghbrs)] = 1
	scale = pre * rich[vals] / np.mean(rich[vals])

	if gal:
		ells = [Ellipse(xy=[l[vals][i], b[vals][i]], width=scale[i], height=scale[i]*0.25, \
			angle=theta[i], alpha=0.5) for i in range(len(theta))]
	else:
		ells = [Ellipse(xy=[ra[vals][i], dec[vals][i]], width=scale[i], height=scale[i]*0.25, \
			angle=theta[i], alpha=0.5) for i in range(len(theta))]


	fig = plt.figure()
	ax = fig.add_subplot(111, aspect='equal')
	for e in ells:
		ax.add_artist(e)
		ax.set_clip_box(ax.bbox)
	if gal:
		ax.set_xlim(min(l[vals]), max(l[vals]))
		ax.set_ylim(min(b[vals]), max(b[vals]))
	else:	
		ax.set_xlim(min(ra[vals]), max(ra[vals]))
		ax.set_ylim(min(dec[vals]), max(dec[vals]))
	ax.set_title("spec={}; sep={}; {:.3f}<z<{:.3f}".format(spec, sep, z_bins[bin_num-1], z_bins[bin_num]))
	ax.set_ylabel("lat[deg]")
	ax.set_xlabel("lon [deg]")
	fig.savefig("redmapper_theta_{}".format(name), dpi=500)
	#fig.show()


	


