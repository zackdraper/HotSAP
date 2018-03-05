#! /usr/bin/env python
import SRP

import sys
import os

import pyfits
import numpy as np
import pandas as pd
import subprocess
import threading
import shlex

import pickle
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.signal import wiener
from scipy.signal import butter
from scipy.optimize import curve_fit
import scipy.ndimage.filters as filt
from scipy.interpolate import interp1d 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from scipy.interpolate import interp2d 
from scipy.interpolate import bisplrep
from scipy.interpolate import bisplev
from scipy.interpolate import SmoothBivariateSpline as sbs
from scipy.integrate import simps
from scipy import optimize
from matplotlib.patches import Ellipse
import time
import datetime
from astroquery.simbad import Simbad
import astropy.units as u
from astropy import coordinates
import astropy.time.core
import astropy
import glob
import shutil
import csv

if __name__ == "__main__":

###########################################################################################################################################################
# HOT Stellar Abundances Pipeline 

	data_dir = 'Data/'

	output_dir = 'pipe_output/'

	cntnorm_dir = output_dir+'Continuum_Normalized/' 

	plot_dir = output_dir+'Stellar_Parameter_Plots/'

	ew_dir = output_dir+'Metal_Line_EW/'

	bal_dir = output_dir+'Balmer_Line/'

	cal_dir = output_dir+'Calibrations/'

	temp_dir = output_dir+'Temp/'


###########################################################################################################################################################

#Make directories if not present

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	if not os.path.exists(cntnorm_dir):
		os.makedirs(cntnorm_dir)

	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)	

	if not os.path.exists(ew_dir):
		os.makedirs(ew_dir)

	if not os.path.exists(bal_dir):
		os.makedirs(bal_dir)

	if not os.path.exists(cal_dir):
		os.makedirs(cal_dir)	

	if not os.path.exists(temp_dir):
		os.makedirs(temp_dir)

	files = os.listdir(data_dir+'.')
	files = np.append(files,os.listdir(cal_dir+'.'))
	files = np.append(files,os.listdir(temp_dir+'.'))
	flats=[]
	dark=[]
	lamp=[]
	obj=[]
	other=[]
	names=[]

	timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
	
#read in SED temperature data

	#print "Importing DEBRIS"
	#csv_data = open('../DEBRIS/SEDs-fluxes-24Apr14-v7.0.csv','r')
	#data = list(csv.reader(csv_data))
	#debris = dict( zip(data[0] , SRP.get_mulList(*data[1:]) ))
	#sed_teff = debris["teff"]
	#star_ids = debris["ID"]

	#catalog = pickle.load(open("../DEBRIS/DEBRIS_astropy_catalog.pickle","r"))
	#pipe_table=[]
	

#mark bad files

	#files = list(set(files)-set(["dl60232.fits","dl60233.fits","dl60234.fits"]))


#data sorting

	#sort files by basic types and target names
	for i in files:
		if ('.fits' in i):
			header=pyfits.getheader(data_dir+i)
			try:
				typ = header['IMAGETYP']	
			except KeyError:
				typ = "other"

			if (typ == "other"):
				try:
					typ = header['EXPTYPE']	
				except KeyError:
					typ = "other"
			typ = typ.lower()

			if 'flat' in typ:
				flats.append(i)
			elif 'dark' in typ:
				dark.append(i)
			elif 'comp' in typ:
				lamp.append(i)
			elif 'object' in typ:
				obj.append(i)

				try:
					name = header['OBJECT']
					
				except (KeyError):
					ra = header['RA']
					dec = header['DEC']
					c = coordinates.SkyCoord(ra+' '+dec,frame='fk4',unit=(u.hourangle, u.deg), obstime="J2000")
					resultTable = Simbad.query_region(c,radius=5*u.arcminute)
					simbad_star_name = resultTable["MAIN_ID"][0]
					resultTable2 = Simbad.query_objectids(simbad_star_name)
					for id_name in resultTable2:
						if 'HR ' in id_name[0]:
							name = id_name[0]
							 

				names.append(name)
			elif True:
				other.append(i)


	#sort object types
	names_u=list(set(names))
	targets={}
	for i in names_u:
		fn=[]
		for v,j in enumerate(names):
			if i == j:
				fn.append(obj[v])
		targets[i.replace(" ","")]=fn


	#clean out the idiotic yet somehow nessacary database
	if (os.path.isdir('./database')):
		files_database = os.listdir('./database')
		for i in files_database:
			os.remove('./database/'+i)

	print "Objects in data set:  "
	for t,v in targets.iteritems():
		print t


#target processing

	#extract spectrum from each star

	beastmode=raw_input('Beast Mode ? (y/n): ').lower()
	beast = beastmode in "y"

	findstar=raw_input('Looking for a star in particular ? (star catalog id): ').lower()
	
	targets_new = dict(targets)

	if findstar == "n":
		print "Looping through all stars"
	else:
		for i,v in targets.iteritems():
			if (findstar in i.lower()):
				print 'Found ',i 
			else:
				del targets_new[i]

	start = time.time()

	for i,v in targets_new.iteritems():

		#print v
		v=np.array(v)

		if (str(i) in [] ):
			print "Skipping: "+str(i)
			continue
		
#spectral analysis
		if(beast):
			proc2 = "y"
		else:
			proc2=raw_input('Proceed with analysis of spectrum '+i+'? (y/n): ').lower()
		

		if proc2 in "y":
			star = str(i)

			star = star.replace('(','_')
			star = star.replace(')','')

			star_file = star.replace('*','')+'.fits'
			str_star = star_file.replace(".fits","_cntnorm_legendre.fits")
	
			star_hdu1 = pyfits.open(cntnorm_dir+str_star.replace("[0]",""))
			star_data1 = star_hdu1[0].data
			star_hdr1 = star_hdu1[0].header
			star_hdu1.close()

			if (np.shape(star_data1)[0] > 2):
				rvcorr_lamb = star_data1[0:,0]
				flux = star_data1[0:,1]
			else:
				rvcorr_lamb = star_data1[0,0:]
				flux = star_data1[1,0:]

			str_star = star_file.replace(".fits","_cntnorm_twod.fits")
	
			star_hdu2 = pyfits.open(cntnorm_dir+str_star.replace("[0]",""))
			star_data2 = star_hdu2[0].data
			star_hdr2 = star_hdu2[0].header
			star_hdu2.close()

			if (np.shape(star_data2)[0] > 2):
				rvcorr_lamb_balmer = star_data2[0:,0]
				flux_balmer = star_data2[0:,1]
			else:
				rvcorr_lamb_balmer = star_data2[0,0:]
				flux_balmer = star_data2[1,0:]

		#find metal lines

			#Venn 1990
			linelist = [
				# Mg I
				["Mg1",4702.991,4.346,-0.67],
				["Mg1",5183.604,2.717,-0.18],
				["Mg1",5711.088,4.346,-1.83],
				["Mg1",5528.405,4.346,-0.62],

				# Mg II

				["Mg2",4242.448,11.56,-1.07],
				["Mg2",4242.542,11.56,-1.23],
				["Mg2",4427.994,9.995,-1.21],
				["Mg2",4481.126,8.863,0.74],
				["Mg2",4481.150,8.863,-0.56],
				["Mg2",4481.325,8.864,0.59],

				# Fe I
				["Fe1",3920.26,0.120,-1.75],
				#["Fe1",3930.30,0.090,-1.59],
				["Fe1",3997.39,2.730,-0.39],
				["Fe1",4005.25,1.560,-0.61],
				["Fe1",4045.82,1.480,0.28],
				["Fe1",4063.60,1.560,-0.61],
				["Fe1",4071.74,1.610,-0.02],
				["Fe1",4202.03,1.480,-0.71],
				["Fe1",4235.94,2.430,-0.34],
				["Fe1",4260.48,2.400,0.02],
				["Fe1",4282.41,2.180,-0.81],
				["Fe1",4466.55,2.830,-0.59],

				# Fe II
				["Fe2",3922.91,7.516,-1.20],
				["Fe2",4233.17,2.580,-2.00],
				["Fe2",4472.92,2.860,-2.21],
				["Fe2",4508.28,2.860,-2.21],
				["Fe2",4515.34,2.840,-2.48],
				["Fe2",4520.23,2.810,-2.60],
				["Fe2",4522.63,2.840,-2.03],
				["Fe2",4541.52,2.860,-3.05],
				["Fe2",4576.33,2.840,-3.04],
				["Fe2",4582.84,2.840,-3.10],
				["Fe2",4583.83,2.810,-2.02],
				["Fe2",4629.34,2.810,-2.37]
			];

			linelist = np.array(linelist)

		#get EW measurements

			EW_data = np.array([0,0,0])
			dw = 5
			sz = np.shape(linelist)[0]
			wlist = (linelist[0:,1]).astype(np.float)
			eVlist = (linelist[0:,2]).astype(np.float)

			for tag,l in enumerate(linelist):
	
				w = np.float(l[1])

				# boundary exception
				#if (np.abs(tag - np.floor(sz/2)) < np.floor(sz/2)):
				bid = np.where( (wlist > (w-dw)) & (wlist < (w+dw)) )
				if (np.size(bid) == 1):
					mtype = "singlet"
				elif (np.size(bid) == 2):
					mtype = "doublet"
				elif (np.size(bid) == 3):
					mtype = "triplet"
				#sys.exit()
							

				ids = np.where((rvcorr_lamb < w+dw) & (rvcorr_lamb > w-dw))

				number = np.shape(ids)[1]
				#weights = np.sqrt(np.abs( np.arange(number)-np.floor(number/2) )+1)/number	

				#need to get fancy to renormalize the metal lines so that gaussian fit works

				linewave = rvcorr_lamb[ids]
				lineflux = flux[ids]

				#recenter around nearby line
				toff = SRP.find_line(linewave,lineflux,w,4)-w
				w2 = w+toff
				
				ids = np.where((rvcorr_lamb < (w2+dw)) & (rvcorr_lamb > (w2-dw)))
				linewave = rvcorr_lamb[ids]
				lineflux = flux[ids]

				#midwave = np.int(np.shape(linewave)[0]/2)
				#res = np.abs(linewave[midwave]-linewave[midwave+1])
				
				tab_p_prime = lineflux-np.roll((lineflux ),3)
				tab_m_prime = np.roll((lineflux),-3)-lineflux

				tab_p_prime2 = lineflux-np.roll((lineflux),7)
				tab_m_prime2 = np.roll((lineflux),-7)-lineflux

				condition = (tab_p_prime > 0) & (tab_m_prime < 0) & (tab_p_prime2 > 0) & (tab_m_prime2 < 0)

				high_points = lineflux[condition]
				hpts_wave = linewave[condition]

				#no highpoints found then skip
				if (np.size(hpts_wave) == 0):
					break

				fit = np.polyfit(hpts_wave,high_points,1)
				norm = linewave*fit[0]+fit[1]

				lineflux = lineflux/norm
				high_points = high_points/norm[condition]

				#iterate local fit
				
				m_high_points = np.median(high_points)
				
				cond = np.abs(lineflux-m_high_points) < 0.01
				upper_high_points = lineflux[cond]
				hpts_wave = linewave[cond]

				#no highpoints found then skip
				if (np.size(hpts_wave) == 0):
					break

				fit = np.polyfit(hpts_wave,upper_high_points,1)
				norm = linewave*fit[0]+fit[1]

				lineflux = lineflux/norm
				high_points = upper_high_points/norm[cond]

				red_hpts_wave = hpts_wave[hpts_wave > w2]
				blu_hpts_wave = hpts_wave[hpts_wave < w2]

				inner_red = np.min(np.append(red_hpts_wave,np.max(linewave)))
				inner_blu = np.max(np.append(blu_hpts_wave,np.min(linewave)))

				f_min_red = lineflux[linewave == inner_red]
				f_min_blue = lineflux[linewave == inner_blu]

				red_upper_high_points = upper_high_points[hpts_wave > w2]
				blu_upper_high_points = upper_high_points[hpts_wave < w2]

				high_points = np.append( blu_upper_high_points[blu_upper_high_points > f_min_blue],red_upper_high_points[red_upper_high_points > f_min_red] )
				hpts_wave = np.append( blu_hpts_wave[blu_upper_high_points > f_min_blue],red_hpts_wave[red_upper_high_points > f_min_red] )

				if (np.size(hpts_wave) > 4):
					fixrange = [np.min(hpts_wave),np.max(hpts_wave)]

					if (np.shape(red_upper_high_points)[0] < 1):
						fixrange[1] = np.max(linewave)
					if (np.shape(blu_upper_high_points)[0] < 1):
						fixrange[0] = np.min(linewave)
				else:
					fixrange = [np.min(linewave),np.max(linewave)]


				lineflux2 = lineflux[(linewave > fixrange[0]) & (linewave < fixrange[1])]
				linewave2 = linewave[(linewave > fixrange[0]) & (linewave < fixrange[1])]

				lineflux2 = pd.rolling_mean(lineflux2,window=10,min_periods=3,center=True)

				#do Gaussian fit to get EW
				model = np.ones(np.size(linewave2))	

				try:
					if (mtype == "singlet"):

						popt = [-0.05, np.mean(linewave2), 1, 1]
						#force gaussian fit to be practical
						# allow it to be impractical to weed out bad lines?
						bounds = ([-1,-np.inf,0,1.0],[0,np.inf,dw/2,1.4])

						popt, pcov = curve_fit(SRP.gauss_function, linewave2, lineflux2, p0 = popt, bounds = bounds)
						err = np.diagonal(pcov)

						model = SRP.gauss_function(linewave2, *popt)

					if (mtype == "doublet"):

						del_wav = np.abs( np.diff(wlist[bid]) )[0]
						del_eV = eVlist[bid]

						#print del_eV
						del_eV = (np.sum(del_eV)/del_eV)[0]
						#print del_eV

						w = np.mean(linewave2)

						toff = SRP.find_line(linewave2,lineflux2,w,3)-w

						popt = [(-0.2/del_eV), w+toff, 0.9, 1, (del_eV-1), del_wav]
						#x, a, m, sigma, x0, a_2, dm_2
						bounds = ([-1,np.min(linewave2),0,1.0,(del_eV-1)*0.9,del_wav*0.8],[-0.01,np.max(linewave2),1.5,2.0,(del_eV-1)*1.1,del_wav*1.2])

						popt, pcov = curve_fit(SRP.gauss_function_2, linewave2, lineflux2, p0 = popt, bounds = bounds)
						err = np.diagonal(pcov)

						model = SRP.gauss_function_2(linewave2, *popt)

					if (mtype == "triplet"):

						del_wav = np.diff(np.abs(wlist[bid]))

						popt = [-0.1, np.mean(linewave2), 1, 1, -0.05, del_wav[0], -0.05, del_wav[1]]

						bounds = ([-1,-np.inf,0,1.0,-1,del_wav[0]*0.5,-1,del_wav[1]*0.9],[0,np.inf,dw/2,1.4,0,del_wav[0]*1.5,0,del_wav[1]*1.1])

						popt, pcov = curve_fit(SRP.gauss_function_3, linewave2, lineflux2, p0 = popt, bounds = bounds)
						err = np.diagonal(pcov)

						model = SRP.gauss_function_3(linewave2, *popt)
				

				except (TypeError,IndexError,ValueError,OSError):
					continue
				
				EW = np.abs(popt[0]*np.sqrt(2*np.pi*(popt[2]*1000)**2))#in milli-Angstroms
				EWerr = (2*np.pi)**2*popt[0]*popt[2]*err[0]*err[2]

				goodnessoffit = np.sum(err)

				EW_data = np.vstack((EW_data,[EW,EWerr,goodnessoffit]))

				plt.clf()
				plt.axvline(fixrange[0],0,2,color='k')
				plt.axvline(fixrange[1],0,2,color='k')
				plt.axvline(l[1],0,2,color='k',ls='--')
				plt.plot(linewave, lineflux)
				plt.plot(linewave2, model)

				if (mtype == "doublet"):
					print w,popt
					EW = np.abs(popt[0]*np.sqrt(2*np.pi*(popt[2]*1000)**2))
					EW = np.abs(popt[0]*np.sqrt(2*np.pi*(popt[2]*1000)**2))
					plt.plot(linewave2,SRP.gauss_function(linewave2,popt[0],popt[1],popt[2],popt[3]),'g--')
					plt.plot(linewave2,SRP.gauss_function(linewave2,(popt[0]*popt[4]),(popt[1]+popt[5]),popt[2],popt[3]),'g--')
					#sys.exit()

				plt.plot(hpts_wave, high_points, 'ro')
				plt.plot(linewave, norm, 'c-.')
				plt.xlim([w2-dw,w2+dw])
				plt.title(str(l[0])+" : "+str(w))

				star_dir = ew_dir+star.replace('*','')+'/'

				if not os.path.exists(star_dir):
					os.makedirs(star_dir)

				plt.savefig(star_dir+str(l[0])+'_'+str(l[1])+'_'+str(tag)+'.pdf')
				#plt.pause(1)


			#sys.exit()
			EW_data = EW_data[1:,0:]
			
			#cut out bad fits by relative goodness of fits
			#cut out fits or low EW and likely bad
			med = np.median(EW_data[0:,2])
			ids = np.where((EW_data[0:,2] < 3*med) & (EW_data[0:,0] > 10))

			linelist = np.array(linelist)

			# assuming element is a string and converting to a number for larger array output
			output = np.array([0,0,0,0,0,0,0],dtype=np.float)
			for m in ids[0]:
				output = np.vstack((output,[np.float(SRP.elementnumber(linelist[m,0])),np.float(linelist[m,1]),np.float(linelist[m,2]),np.float(linelist[m,3]),EW_data[m,0],EW_data[m,1],EW_data[m,2]]))

			output = output[1:,0:]

			#sys.exit()
			#find triplets and doublets to distribute EW values

			#lines = output[0:,1]
			#for e,o in enumerate(output):
				#line_w = o[1]
				#ids = np.where(np.abs(lines-line_w)<dw)
				#num = np.shape(ids)[1]
				#print num
				#if (num > 1):
					#output[e,4] = output[e,4]/num

			

		#prep for pyBLAMER 9

			if ('*' in star):
				star = star.replace("*","")

			moogfile = SRP.writetomoog(temp_dir+star+"_temp",output)

			#znew3,x,y = pybalmer9(moogfile,[rvcorr_lamb,flux],star)
			#def pybalmer9(linelist,data,star):
			data = [rvcorr_lamb_balmer,flux_balmer]

		#main code balmer 9

			interp_x = 29
			interp_y = 11
			interp_kind = 'linear'

			dats_dir = os.listdir('./DATS/')

			datfiles_mmod = []
			for x in dats_dir: 
				if 'mmod' in x:
					#print x
					datfiles_mmod.append([x])

			# model and match blamer lines

			dats_dir = os.listdir('./DATS/')

			datfiles_kmod = []
			for x in dats_dir: 
				if 'kmod' in x:
					#print x
					datfiles_kmod.append([x])

			models_dist = []

			star_dir = bal_dir+star.replace('*','')+'/'
			if not os.path.exists(star_dir):
				os.makedirs(star_dir)	

			first = 0
			for x in datfiles_kmod:
				SRP.execute_balmer9('./DATS/'+str(x[0]))
	
				chisqr,beta_line,gamma_line,delta_line = SRP.chisqr_balmer9(data)
				
				modelname = str(x[0])
				modelname = modelname.replace("['","")
				modelname = modelname.replace("']","")
			
				if (('g450' in modelname) or ('T9000' in modelname)):
					plt.clf()
					plt.title("H Beta")
					plt.plot(beta_line[0],beta_line[2],'r')
					plt.plot(beta_line[0],beta_line[1],'b')
					plt.plot(beta_line[0][beta_line[3]],beta_line[1][beta_line[3]],'go')
					plt.xlim([4800,4920])
					plt.ylabel("Normalized Intensity")
					plt.xlabel("Wavelength (Angs)")
					plt.savefig(star_dir+'hbeta_'+str(x)+'.pdf')
	
					plt.clf()
					plt.title("H Gamma")
					plt.plot(gamma_line[0],gamma_line[2],'r')
					plt.plot(gamma_line[0],gamma_line[1],'b')
					plt.plot(gamma_line[0][gamma_line[3]],gamma_line[1][gamma_line[3]],'go')
					plt.xlim([4280,4400])
					plt.ylabel("Normalized Intensity")
					plt.xlabel("Wavelength (Angs)")
					plt.savefig(star_dir+'hgamma_'+str(x)+'.pdf')

					plt.clf()
					plt.title("H Delta")
					plt.plot(delta_line[0],delta_line[2],'r')
					plt.plot(delta_line[0],delta_line[1],'b')
					plt.plot(delta_line[0][delta_line[3]],delta_line[1][delta_line[3]],'go')
					plt.xlim([4040,4160])
					plt.ylabel("Normalized Intensity")
					plt.xlabel("Wavelength (Angs)")
					plt.savefig(star_dir+'hdelta_'+str(x)+'.pdf')

				f=open('./auto.dat','r')
				model_info = f.readline()
				model_info = filter(None, model_info.split(" "))
				teff = model_info[1]
				logg = model_info[3]
							
				if (first == 0):
					chi_stop = np.array([True,True,True],dtype=bool)
				
					# ignore a line if there is a large discontinuity
					beta_line_med = medfilt(beta_line[1],kernel_size=41)
					if ( np.max((beta_line_med - np.roll(beta_line_med,-1))) > 0.1):
						chi_stop[0] = False
					delta_line_med = medfilt(delta_line[1],kernel_size=41)
					if ( np.max((delta_line_med - np.roll(delta_line_med,-1))) > 0.1):
						chi_stop[1] = False
					gamma_line_med = medfilt(gamma_line[1],kernel_size=41)
					if ( np.max((gamma_line_med - np.roll(gamma_line_med,-1))) > 0.1):
						chi_stop[2] = False
					first = 1
					
					if (np.sum(chi_stop) == 0):
						chi_stop = np.array([True,True,True],dtype=bool)	
				
	
				update = [float(teff),float(logg),float(np.sum(chisqr[chi_stop]))]
				#print update
				if not np.isnan(np.sum(chisqr[chi_stop])):
					models_dist.append(update)


			tst2=np.array(models_dist)
			x =  np.linspace(min(tst2[:,0]),max(tst2[:,0]),interp_x)
			y =  np.linspace(min(tst2[:,1]),max(tst2[:,1]),interp_y)

			znew2 = np.zeros(shape=(interp_x,interp_y), dtype=np.float)

			for m,i in enumerate(x):
				for n,j in enumerate(y):
					id1 = np.in1d(tst2[:,0],i)
					id2 = np.in1d(tst2[:,1],j)
					match = id1*id2
					if np.sum(match) > 0:
						znew2[m,n] = 1.0/tst2[match,2]


			#mask balmer line fits

			#mask = np.ones(np.shape(znew2))
			#znew2 = znew2*mask

			mv = np.nanmax(znew2[0:,0:])

			for m,i in enumerate(x):
				for n,j in enumerate(y):
					znew2[m,n] = znew2[m,n]/mv

			fig = plt.figure()
			p = plt.pcolor(x,y,np.transpose(znew2),vmax=1.0,vmin=0.0)
			plt.colorbar()
			plt.ylabel('log(g)')
			plt.xlabel(r'$T_{eff}$')
			plt.title('Balmer Line Fitting')
			plt.savefig(plot_dir+star+"_blf.pdf")

			# end balmer line matching

			# compare abundaces

			models_dist2 = []
			for x in datfiles_mmod:
				SRP.create_moog_file('./DATS/'+str(x[0]),moogfile)
		
				key,a = SRP.execute_moog(ew_dir,star)
				while (np.size(a) < 6):
					a=np.append(a,[0.0])
			
	
				if not np.isnan(np.sum(a)):
					#print a
					mg = (np.abs((a[key == 'MgI']-a[key == 'MgII'])))
					fe = (np.abs((a[key == 'FeI']-a[key == 'FeII'])))

					try:
						mg = float(mg[0])
					except (IndexError):
						mg = 99

					try:
						fe = float(fe[0])
					except (IndexError):
						fe = 99
		
					update = [float(a[0]),float(a[1]),fe,mg]
					#print update
					models_dist2.append(update)


			tst1=np.array(models_dist2)
			print np.array(sorted(models_dist2, key =lambda models_dist2: models_dist2[0]))

			znew = np.zeros(shape=(interp_x,interp_y), dtype=np.float)

			x = np.linspace(min(tst2[:,0]),max(tst2[:,0]),interp_x)
			y = np.linspace(min(tst2[:,1]),max(tst2[:,1]),interp_y)


			for m,i in enumerate(x):
				for n,j in enumerate(y):
					#znew[m,n] = 1/tck.ev(i, j)
					id1 = np.in1d(tst1[:,0],i)
					id2 = np.in1d(tst1[:,1],j)
					match = id1*id2
					if np.sum(match) > 0:
						znew[m,n] = tst1[match,2]+1

			znew = 1/znew
			znew[znew == np.inf] = 0.0
			znew = znew/np.max(znew)
			#znew = filt.median_filter(znew,size=(3,3),mode="nearest")

			teff_1 = []
			logg_1 = []
			for i,logg in enumerate(y):
				min_id = np.where(znew[:,i] == np.nanmax(znew[:,i]))

				min_v = np.mean(x[min_id])

				#print min_v

				logg_1 = np.append(logg_1,logg)
				teff_1 = np.append(teff_1,min_v)

			#print 1.0/(tst1[:,2]+1.0)
	
			fig = plt.figure()
			p = plt.pcolor(x,y,np.transpose(znew),vmax=1.0, vmin=0.0)
			plt.colorbar()
			plt.ylabel('log(g)')
			plt.xlabel(r'$T_{eff}$')
			plt.title('Abundance Comparison')
			plt.savefig(plot_dir+star+"_abund.pdf")

			#end abundance matching

			fig = plt.figure()
			teff_2 = np.array([])
			logg_2 = np.array([])

			for i,logg in enumerate(y):
				max_id = np.where(znew2[:,i] == np.nanmax(znew2[:,i]))

				min_v = np.mean(x[max_id])

				logg_2 = np.append(logg_2,logg)
				teff_2 = np.append(teff_2,min_v)

			plt.plot(teff_2,logg_2,'b',label="Balmer Line")
			plt.plot(teff_1,logg_1,'g',label="Double Ionzied Species")
			plt.ylabel('log(g)')
			plt.xlabel(r'$T_{eff}$')

			#clean up nans
			teff_1 = np.nan_to_num(teff_1)
			teff_2 = np.nan_to_num(teff_2)

			fit1 = np.polyfit(teff_1,logg_1,1)
			fit2 = np.polyfit(teff_2,logg_2,1)

			plt.plot(teff_1,np.array(teff_1)*fit1[0]+fit1[1],'-r')
			plt.plot(teff_2,np.array(teff_2)*fit2[0]+fit2[1],'-r')

			plt.xlim([min(x),max(x)])
			plt.ylim([min(y),max(y)])

			plt.legend(loc=4)
			plt.savefig(plot_dir+star+"_trendlines.pdf")

			x = (fit1[1]-fit2[1])/(fit2[0]-fit1[0])
			y = fit2[0]*x+fit2[1]
			#print x,y

			match_model_dist = []
			for t1 in tst1: 
				for t2 in tst2: 
					if ((t1[0] == t2[0]) & (t1[1] == t2[1])):
						update = [t1[0],t2[1],t2[2],t1[2],t1[3]]
						#print update
						match_model_dist.append(update)

			tst3=np.array(match_model_dist)

			stat = ( tst3[:,2]/sum(tst3[:,2]) )+( tst3[:,3]/sum(tst3[:,3]) )
			min_stat = np.where(stat == min(stat))

			#print tst3[min_stat,0:2][0][0]

			x =  np.linspace(min(tst1[:,0]),max(tst1[:,0]),interp_x)
			y =  np.linspace(min(tst1[:,1]),max(tst1[:,1]),interp_y)

			znew3 = (znew2)*(znew)

		#end blamer 9


			fig = plt.figure()
			ax = fig.add_subplot(111)

			sz = np.shape(znew3)
			
			p = plt.pcolor(x,y,np.transpose(znew3),vmax=1.0,vmin=0.0)
			plt.colorbar()
			plt.ylabel('log(g)')
			plt.xlabel(r'$T_{eff}$')
			plt.title('Combined Prob')

			znew3[np.isnan(znew3)] = 0

			#znew3 = np.reshape(znew3, (np.size(x),np.size(y)) )
			#print np.shape(znew3)

			gridx,gridy = np.meshgrid(x,y)

			#popt = (1.0,np.median(x),np.median(y),500,1.0,0.0,0.0)

			#popt, pcov = curve_fit(gauss_function_2d, (x,y), znew3, p0 = popt, bounds = bounds)
			
			maxid = np.where(znew3 == np.max(znew3))

			p = SRP.heat_map_fit(znew3,x,y,gridx,gridy)
				
			e = Ellipse(xy = (p[1],p[2]), width = p[3]/2, height = p[4]/2, angle=0, linewidth=2, fill=False)
			ax.add_artist(e)
			plt.plot(p[1],p[2],'k*')

			plt.savefig(plot_dir+star+"_cbp.pdf")
						

			star_param = np.array([])
			try:
				f = open( output_dir+"pipe_table_"+timestamp+".pickle", "rb" )
				star_param = pickle.load(f)
				f.close()

			except IOError:
				star_param = np.array([])

			# Get header information
			#ra = star_hdr['RA']
			#dec = star_hdr['DEC']

			#print star
			s = star
			s = s.replace('hd','HD ')
			s = s.replace('Debris','')
			s = s.replace('Debrois','')
			s = s.replace('DEebris','')
			s = s.replace('-Supergiant','')
			s = s.replace('SRd','')
			s = s.replace('-Supergiant','')
			s = s.replace('O','')
			s = s.replace('Her',' Her')
			s = s.replace('KV','')
			s = s.replace('_North','')
			s = s.replace('_South','')
			    
			resultTable2 = Simbad.query_object(s)
			    
			try:
				ra = (resultTable2["RA"][0]).replace(' ',':')
				dec = (resultTable2["DEC"][0]).replace(' ',':')
			except:
				ra = "00:00:00.00"
				dec = "+00:00:00.00"


			#bias by debris survey
			#c = coordinates.SkyCoord(ra+' '+dec,frame='fk4',unit=(u.hourangle, u.deg), obstime="J2000")
    			#id,d2d,d3d = coordinates.match_coordinates_sky(c,catalog)
   			#print id,d2d,d3d
    			#debris data
    			#debris_teff = np.float(sed_teff[id])

			tstrom = SRP.strom_phot(s)

			#znew4 = np.zeros(shape=(interp_x,interp_y), dtype=np.float)
			#znew4[0:,0:] = 0.25
			#dteff=x[1]-x[0]
			#tempid = (x > debris_teff-2*dteff) & (x < debris_teff+2*dteff) 
			#znew4[tempid,0:] = 0.5
			#tempid = (x > debris_teff-dteff) & (x < debris_teff+dteff) 
			#znew4[tempid,0:] = 1.0

			znew5 = np.zeros(shape=(interp_x,interp_y), dtype=np.float)
			znew5[0:,0:] = 0.25
			dteff=x[1]-x[0]
			tempid = (x > tstrom-2*dteff) & (x < tstrom+2*dteff) 
			znew5[tempid,0:] = 0.5
			tempid = (x > tstrom-dteff) & (x < tstrom+dteff) 
			znew5[tempid,0:] = 1.0

			#znew_debris = znew3 * znew4
			znew_strom = znew3 * znew5

			#p_debris = SRP.heat_map_fit(znew_debris,x,y,gridx,gridy)
			p_debris = [0,0,0,0,0]

			#fig = plt.figure()
			#ax = fig.add_subplot(111)

			#e = Ellipse(xy = (p_debris[1],p_debris[2]), width = p_debris[3]/2, height = p_debris[4]/2, angle=0, linewidth=2, fill=False)
			#ax.add_artist(e)
			#plt.plot(p_debris[1],p_debris[2],'k*')

			#ptc = plt.pcolor(x,y,np.transpose(znew_debris),vmax=np.max(znew_debris),vmin=0.0)
			#plt.ylabel('log(g)')
			#plt.xlabel(r'$T_{eff}$')
			#plt.title('DEBRIS & Spec')
			#plt.savefig(plot_dir+star+"_debris.pdf")

			p_strom = SRP.heat_map_fit(znew_strom,x,y,gridx,gridy)

			fig = plt.figure()
			ax = fig.add_subplot(111)

			e = Ellipse(xy = (p_strom[1],p_strom[2]), width = p_strom[3]/2, height = p_strom[4]/2, angle=0, linewidth=2, fill=False)
			ax.add_artist(e)
			plt.plot(p_strom[1],p_strom[2],'k*')

			ptc = plt.pcolor(x,y,np.transpose(znew_strom),vmax=np.max(znew_strom),vmin=0.0)
			plt.ylabel('log(g)')
			plt.xlabel(r'$T_{eff}$')
			plt.title('DEBRIS & Spec')
			plt.savefig(plot_dir+star+"_strom.pdf")

			ind = (-znew_strom).argpartition(3,axis=None)[:3]
			xid,yid = np.unravel_index(ind, znew5.shape)
			teff_max = np.mean(x[xid])
			logg_max = np.mean(y[yid])

			try:
				star_param = np.vstack((star_param,[ [star,p[1],p[3]/2,p[2],p[4]/2,p_debris[1],p_debris[3]/2,p_debris[2],p_debris[4]/2,p_strom[1],p_strom[3]/2,p_strom[2],p_strom[4]/2,teff_max,logg_max] ] ))
			except ValueError:
				star_param = [[star,p[1],p[3]/2,p[2],p[4]/2,p_debris[1],p_debris[3]/2,p_debris[2],p_debris[4]/2,p_strom[1],p_strom[3]/2,p_strom[2],p_strom[4]/2,teff_max,logg_max]]

				
			
			f = open(output_dir+"pipe_table_"+timestamp+".pickle", "wb")
			pickle.dump(star_param, f)
			f.close()


		elif proc in "n":
			print "Skipping star"
		elif proc in "b":
			break
		else:
			print "Bugged input"		
	
		end = time.time()

		print "elapsed time for star: ",np.int(end-start)," sec"


#clean up
	#os.remove('master_dark.fits')

	

		


			
