#! /usr/bin/env python
import SRP

import os
import numpy as np
import datetime
import time
import csv
import pickle
import pyfits
from astropy import coordinates
import astropy.units as u
from astroquery.simbad import Simbad
from pyraf import iraf
import glob
import shutil
import scipy.ndimage.filters as filt
from scipy.signal import medfilt

import matplotlib.pyplot as plt

if __name__ == "__main__":

###########################################################################################################################################################
# Hot-star Stellar Abundances Pipeline 

	data_dir = 'McD_raw_data/'

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
	

#mark bad files

	files = list(set(files)-set(["dl60232.fits","dl60233.fits","dl60234.fits"]))


#data sorting

	#sort files by basic types and target names
	for i in files:
		if '.fits' in i and 'temp' not in i and 'tmp' not in i and 'master' not in i and 'ec' not in i and 'tnrm' not in i and 'HD' not in i and 'KV' not in i and 'dl' in i:
			header=pyfits.getheader(data_dir+i)
			try:
				typ = header['IMAGETYP']
			except KeyError:
				typ = "other"
			
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

	if len(other) > 0: 
		print '*.FITS FILES NOT USED IN REDUCTION '+str(other)

	dataset = flats+dark+lamp+obj

	#read-only protect original data.
	#for i in dataset:
    		#subprocess.call(['chmod', '-v', '777', i])
		#iraf.hedit(i+'[0]',"DISPAXIS",1,verify="no",add="yes")
		#subprocess.call(['chmod', '-v', '444', i])


	#sort object types
	names_u=list(set(names))
	targets={}
	for i in names_u:
		fn=[]
		for v,j in enumerate(names):
			if i == j:
				fn.append(obj[v])
		targets[i.replace(" ","")]=fn

#load iraf packages and selected parameters
	iraf.noao(_doprint=0)
	iraf.onedspec(_doprint=0)
	iraf.twodspec(_doprint=0)
	iraf.imred(_doprint=0)
	iraf.ccdred(_doprint=0)
	iraf.apextract(_doprint=0)
	iraf.echelle(_doprint=0)

	iraf.ccdred.instrum = "ccddb$kpno/camera.dat"

	iraf.imarith.unlearn()
	iraf.imarith.verbose = "yes"

	iraf.flatcombine.unlearn()
	#iraf.flatcombine.interactive = "no"
	iraf.flatcombine.combine="average"

	iraf.apextract.unlearn()
	iraf.apextract.database = "database"
	iraf.apextract.dispaxis = 1

	iraf.apflatten.unlearn()
	iraf.apflatten.interactive = "no"
	iraf.apflatten.recenter = "no"
	iraf.apflatten.resize = "no"
	iraf.apflatten.flatten = "yes"
	iraf.apflatten.find = "no"
	iraf.apflatten.trace = "no"

	iraf.apall.unlearn()
	iraf.apall.minsep = 20
	iraf.apall.maxsep = 26
	iraf.apall.width = 5
	iraf.apall.radius = 10
	iraf.apall.thresho = 100
	iraf.apall.line = 1024
	iraf.apall.nsum = 40
	iraf.apall.t_nsum = 10
	iraf.apall.t_step = 1		
	iraf.apall.t_order = 5
	iraf.apall.t_funct = "legendre"
	iraf.apall.t_niterate = 0
	iraf.apall.nfind = 63
	iraf.apall.format = "echelle"
	iraf.apall.interactive = "no"
	iraf.apall.find = "yes"
	iraf.apall.trace = "yes"
	iraf.apall.extract = "yes"
	iraf.apall.recenter = "yes"
	iraf.apall.clean="yes"
	iraf.apall.saturation=1E6

	iraf.apfind.unlearn()
	iraf.apfind.nfind = 63
	iraf.apfind.minsep = 20
	iraf.apfind.maxsep = 26
	iraf.apfind.interactive = "no"
	iraf.apfind.dispaxis = 1

	iraf.aptrace.unlearn()
	iraf.aptrace.order = 5
	iraf.aptrace.function = 'legendre'
	iraf.aptrace.step = 1
	iraf.aptrace.nsum = 20
	iraf.aptrace.line = 1024
	iraf.aptrace.niterate = 0
	iraf.aptrace.nlost = 5

	iraf.apnormalize.unlearn()
	iraf.apnormalize.order = 3
	iraf.apnormalize.line = 1024
	iraf.apnormalize.nsum = 10
	iraf.apnormalize.function = "legendre"
	iraf.apnormalize.find = "no"
	iraf.apnormalize.trace = "no"
	iraf.apnormalize.fittrac = "no"	
	iraf.apnormalize.edit = "no"
	iraf.apnormalize.resize = "no"
	iraf.apnormalize.recenter = "no"
	iraf.apnormalize.interactive = "no"
	iraf.apnormalize.background = "none"

	iraf.ecidentify.unlearn()
	iraf.ecidentify.threshold = 100
	iraf.ecidentify.coordlist = "linelists$thar.dat" 
	iraf.ecidentify.minsep = 15
	iraf.ecidentify.fwidth = 5

	#unfuck PyRAF apnorm 
	iraf.apnorm1.unlearn()
	iraf.apnorm1.background = ")apnormalize.background"
	iraf.apnorm1.skybox = ")apnormalize.skybox"
	iraf.apnorm1.weights = ")apnormalize.weights"
	iraf.apnorm1.pfit = ")apnormalize.pfit"
	iraf.apnorm1.clean = ")apnormalize.clean"
	iraf.apnorm1.saturation = ")apnormalize.saturation"
	iraf.apnorm1.readnoise = ")apnormalize.readnoise"
	iraf.apnorm1.gain = ")apnormalize.gain"
	iraf.apnorm1.lsigma = ")apnormalize.lsigma"
	iraf.apnorm1.usigma = ")apnormalize.usigma"

	#clean out the idiotic yet somehow nessacary database
	try:
		files_database = os.listdir('./database')
		for i in files_database:
			os.remove('./database/'+i)
	except OSError:
		print "No database to delete yet"

#trim all images of bias/overscan, and fix keywords
	
	all_img = list(flats+dark+obj+lamp)
	all_img2 = SRP.outputnames(all_img,'temp')

	#mkcalib = raw_input("Create trimmed temp files (all)? (y/n): ").lower()

	#if "y" in mkcalib:
		#for x,i in enumerate(all_img):
			#iraf.imcopy(i+'[1:2048,1:2048]',all_img2[x])
			#fix DISPAXIS keyword
			#iraf.hedit(all_img2[x]+'[0]',"DISPAXIS",1,verify="no",add="yes")
	#elif "n" in mkcalib:
		#print "Skipping temporary files"

	#os.system("rm *temp*.fits")
	#os.system("rm *tnrm*.fits")

	print "Objects in data set:  "
	for t,v in targets.iteritems():
		print t

#calibration data

	mkcalib = raw_input("Create CCD calibration files? (y/n): ").lower()

	if "y" in mkcalib:
		print "**** create master dark file ****"
		dark_temp=SRP.outputnames(dark,'temp')
		str_darks=data_dir+('[0],'+data_dir).join(dark)+'[0]'

		if 'master_dark.fits' in files:
			os.remove(cal_dir+'master_dark.fits')
			print 'Removed last master dark'
		
		iraf.imcombine(str_darks,cal_dir+'master_dark.fits',combine="median")

		print "**** create master flat ****"
		if 'master_flat.fits' in files:
			os.remove(cal_dir+'master_flat.fits')
			print 'Removed last master flat'

		nflats = np.size(flats)
		if  nflats > 15:
			print "**** too many damn flats ****"
			t_flats = flats[:]
			i=0
			m_flats = []
			while (nflats > 0):
				i = i+1
				num = np.min([10,np.size(t_flats)-1])
				if num == 0:
					break

				s_flats = t_flats[0:num]

				str_flats=data_dir+('[0],'+data_dir).join(s_flats)+'[0]'
				flats_temp=SRP.outputnames(s_flats,'temp')
				str_flats_temp=temp_dir+(','+temp_dir).join(flats_temp)+''	
				
				t_flats = list(set(t_flats) - set(s_flats))
				nflats = nflats-np.size(s_flats)
				
				print "**** subtract dark from flats ****"
				iraf.imarith(str_flats,'-',cal_dir+'master_dark.fits',str_flats_temp)

				print "**** combine flats ****"
				iraf.reset(use_new_imt="no")
				iraf.flpr("0")

				str_flats_temp=temp_dir+('[0],'+temp_dir).join(flats_temp)+'[0]'
				mflatname = temp_dir+'master_flat_'+str(i)+'.fits'

				iraf.imcombine(str_flats_temp,mflatname,combine="sum")
	
				m_flats.append(mflatname)

				print "**** Loop it again because IRAF sucks ****"

			print "**** combine master flats ****"
			
			str_flats_temp=','.join(m_flats)
			iraf.imcombine(str_flats_temp,cal_dir+'master_flat.fits',combine="sum")


		else:
			str_flats=data_dir+('[0],'+data_dir).join(flats)+'[0]'
			flats_temp=SRP.outputnames(flats,'temp')
			str_flats_temp=temp_dir+(','+temp_dir).join(flats_temp)+''

			print "**** subtract dark from flats ****"
			iraf.imarith(str_flats,'-',cal_dir+'master_dark.fits',str_flats_temp)

			print "**** combine flats ****"
			iraf.reset(use_new_imt="no")
			iraf.flpr("0")

			str_flats_temp=temp_dir+('[0],'+temp_dir).join(flats_temp)+'[0]'
			iraf.imcombine(str_flats_temp,cal_dir+'master_flat.fits',combine="sum")

		str_lamps='[0],'.join(lamp)+'[0]'
		lamps_temp=SRP.outputnames(lamp,'temp')
		str_lamps_temp=temp_dir+('[0],'+temp_dir).join(lamps_temp)+'[0]'

		#subtract darks from lamps
		for l,f in enumerate(lamp):
			iraf.imarith(data_dir+f,'-',cal_dir+'master_dark.fits',temp_dir+'master_lamp_'+str(l)+'_temp.fits')

		#wavelength calibration, need to extract a star to get the trace right
		for l,f in enumerate(lamp):
			#should do a time stamp comparison between lamp and science			

			lampfile = 'master_lamp_'+str(l)+'[0][0]'
			lamptemp = 'master_lamp_'+str(l)+'_temp.fits'
		
			#stack all science images pull out a star the get a reliable extraction of the lamp
			for i,v in targets.iteritems(): 
				star = i
				v_mod = SRP.outputnames(np.array(v),'temp')

			print "**** subtract master dark from star images ****"
			#v_mod =SRP.ouputnames(np.array(v),'temp')
			str_v=data_dir+('[0]'+','+data_dir).join(v)+'[0]'
			str_v_mod=temp_dir+(','+temp_dir).join(np.array(v_mod))
			iraf.imarith(str_v,'-',cal_dir+'master_dark.fits',str_v_mod)

			star_file = star.replace('*','')+'.fits'

			print "**** combine star files ****"
			iraf.imcombine(str_v_mod,temp_dir+star_file,combine="sum",reject="sigclip")

			print "**** trace the stars orders ****"
			iraf.apall(temp_dir+star_file,extract='no',nfind=62,interactive="no",find="yes")

			#print "**** flatten spectra ****"
			#v_mod_norm =SRP.ouputnames(v,'tnrm')
			#str_v_mod_flats=temp_dir+(','+temp_dir).join(v_mod_norm)+''
			#for j,x in enumerate(v_mod):
			#	iraf.apflatten(cal_dir+'master_flat.fits',output=temp_dir+v_mod_norm[j],reference=temp_dir+star_file,clobber=True)

			iraf.reset(use_new_imt="no")
			iraf.flpr("0")

			#print "trace the stars orders"
			#iraf.apall(temp_dir+star_file,extract='no',nfind=62,interactive="no",find="yes")

			lampfile2 = lamptemp.replace('_temp.fits','.ec.fits')
			if lampfile2 in files:
				os.remove(cal_dir+lampfile2)
				print 'Removed last master lamp '+str(l)

			#use star orders extract lamp
			str_lamp = lamptemp.replace('_temp.fits','.ec.fits')
			iraf.apall(temp_dir+(lamptemp.replace('.fits','[0][0]')),extract='yes',nfind=62,interactive="no",find="no",trace="no",output=cal_dir+str_lamp,reference=temp_dir+star_file)


		#remove temporary files to save hardrive space
		
		temp_files_del = os.listdir(temp_dir)
		for f in temp_files_del:
			os.remove(temp_dir+f)
			
		#sys.exit()
		

	elif "n" in mkcalib:
		print "Skipping calibration files"
	else:
		print "input bugged" 

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
		
		if(beast):
			proc = "y"
		else:
			proc=raw_input('Proceed with reduction of '+i+'? (y/n): ').lower()

		if proc in "y":

			#Why the fuck does a subsection trimming change x&y coordinates?
			#trimed_ccd = '[1:2047,2:2047]'
			trimed_ccd=''
			str_v=data_dir+('[0]'+trimed_ccd+','+data_dir).join(v)+'[0]'+trimed_ccd
			#print str_v

			#subtract master dark from star images
			v_mod = SRP.outputnames(np.array(v),'temp')
			str_v_mod=temp_dir+(','+temp_dir).join(np.array(v_mod))
			iraf.imarith(str_v,'-',cal_dir+'master_dark.fits'+trimed_ccd,str_v_mod)

			#stack all science images
			star = str(i)
			star = star.replace('(','_')
			star = star.replace(')','')
	
			star_file = star.replace('*','')+'.fits'
			str_v_mod_out=temp_dir+('[0]'+','+temp_dir).join(v_mod)+'[0]'

			if star_file in files:
				os.remove(temp_dir+star_file)
				print 'Removed star file: '+star_file

			iraf.imcombine(str_v_mod_out,temp_dir+star_file,combine="sum")

			#boost SNR with flat to do tracing
			if star_file.replace('.fits','_boost.fits') in files:
				os.remove(temp_dir+star_file.replace('.fits','_boost.fits'))
				print 'Removed star file: '+star_file.replace('.fits','_boost.fits')

			iraf.imarith(cal_dir+'master_flat.fits','*',temp_dir+star_file,temp_dir+star_file.replace('.fits','_boost.fits'))

			iraf.reset(use_new_imt="no")
			iraf.flpr("0")

			#make flats per spectrum tracing, using staked science aptrace
			str_v_mod_out=temp_dir+('[0]'+','+temp_dir).join(v_mod)+'[0]'

			#iraf.apall("master_flat.fits",extract='no',nfind=58,interactive="no",find="yes")

			iraf.apall(temp_dir+star_file.replace('.fits','_boost.fits'),extract='no',nfind=62,interactive="no",find="yes",recenter="yes")
			v_mod_norm =SRP.outputnames(v,'tnrm')
			str_v_mod_flats=temp_dir+(','+temp_dir).join(v_mod_norm)+''
			for j,x in enumerate(v_mod):
				iraf.apflatten(cal_dir+'master_flat.fits',output=temp_dir+v_mod_norm[j],reference=temp_dir+star_file.replace('.fits','_boost.fits'))

			iraf.reset(use_new_imt="no")
			iraf.flpr("0")

			#divide each science image by its flat
			iraf.imarith(str_v_mod_out,'/',str_v_mod_flats,str_v_mod)

			#stack all science images
			iraf.imcombine(str_v_mod_out,temp_dir+star_file,combine="sum")

			iraf.reset(use_new_imt="no")
			iraf.flpr("0")

			str_star = star_file.replace('.fits','.ec.fits')

			if str_star in files:
				os.remove(temp_dir+str_star)
				print 'Removed star file: '+str_star

			iraf.apall(temp_dir+star_file+'[0][0]',find="no",trace="no",recenter="yes",resize="no",extract="yes",interactive="no",output=(temp_dir+str_star),reference=(temp_dir+star_file.replace('.fits','_boost.fits')))

			#check extraction output
			#gwm.window('Star Order 46')
			#iraf.bplot(str_star,aperture=46,band=2)
				
			#flat normalize lamp spectrum
			#if lamptemp in files:
				#os.remove(lamptemp)
				#print 'Removed star file: '+lamptemp

			#iraf.apflatten('master_flat.fits'+trimed_ccd,output=lamptemp,reference=star_file)
			#iraf.imarith(lampfile+'[0]'+trimed_ccd,'/',lamptemp,lamptemp)
			
			#ext_spec = lamptemp.replace('.fits','.ec.fits')
			#if ext_spec in files:
				#os.remove(ext_spec)
				#print 'Removed star file: '+ext_spec
				
			#wavelength calibration assuming a calibration array has been made by my code

			#write a smart way to find the nearest time stamped wave cal
			wcalfile = glob.glob(cal_dir+"disp_arr*.pickle")[0]

			cal_array = pickle.load( open( wcalfile ) )

			calsz = np.shape(cal_array)
						
			shutil.copyfile(temp_dir+str_star.replace("[0]",""),cntnorm_dir+str_star.replace("[0]",""))

			#bring star file into python
			star_hdu = pyfits.open(temp_dir+str_star.replace("[0]",""))
			star_data = (star_hdu[0].data)[1,0:,0:]
			star_hdr = star_hdu[0].header
			star_hdu.close()
			orders = np.shape(star_data)[0]
			pixels = np.shape(star_data)[1]
			
			if (orders == calsz[0]): 
				print "same size"
			else:
				print "mismatch size"
				sys.exit()

	#flatten continuum

	#legendre fitting

			norm_lamb=np.array([])
			norm_flux=np.array([])
			
			x = np.arange(orders,dtype=np.float)		

			pix_s = np.arange(2045,dtype=np.float)

			#BAD ORDERS
			x = x[1:]
			
			
			cont_method = "smooth_orders"
			cont_method = "legendre_fit"

			#cut overscan
			star_data = star_data[0:,1:2046]


			flat = np.array(star_data[0:,0:],dtype=np.float)

			#plt.ion()
			#plt.show()


			for j in x:

				pix = (pix_s*cal_array[j,1])+cal_array[j,2]

		#blend orders to remove broad lines for normalization

				if (j == x[0]):
					star_flux_chunk1 = star_data[j,0:]
					star_flux_chunk2 = star_data[j+1,0:]
					star_flux_chunk = np.median([[star_flux_chunk1],[star_flux_chunk2]],axis=0)[0]
				elif (j == x[-1]):
					star_flux_chunk1 = star_data[j,0:]
					star_flux_chunk2 = star_data[j-1,0:]
					star_flux_chunk = np.median([[star_flux_chunk1],[star_flux_chunk2]],axis=0)[0]
				else:
					star_flux_chunk1 = star_data[j+1,0:]
					star_flux_chunk2 = star_data[j,0:]
					star_flux_chunk3 = star_data[j-1,0:]
					star_flux_chunk = np.median([[star_flux_chunk1],[star_flux_chunk2],[star_flux_chunk3]],axis=0)[0]

				#print np.shape(star_flux_chunk),j
		#filter spectra for continuum via fancy methods

				mt = np.mean(star_flux_chunk)

				tab=np.array(star_flux_chunk)
				scale = max([np.abs(max(tab)-min(tab)),20000])
				tab_prime=((max(tab)-tab)*scale)/max(tab)

				#plt.plot(tab, 'b', hold=True)

				#plt.plot(tab+tab_prime, 'r')
				#plt.ylim(mean(tab)-0.5,mean(tab)+0.5)
				#plt.show()

				sl_corrected = tab+tab_prime

				tab_p_prime = sl_corrected-np.roll((sl_corrected),5)
				tab_m_prime = np.roll((sl_corrected),-5)-sl_corrected

				tab_p_prime2 = sl_corrected-np.roll((sl_corrected),10)
				tab_m_prime2 = np.roll((sl_corrected),-10)-sl_corrected

				condition = (tab_p_prime > 0) & (tab_m_prime < 0) & (tab_p_prime2 > 0) & (tab_m_prime2 < 0)

				high_points = star_flux_chunk[condition]
				wave = pix[condition]

		#strap down at ends

				wave = np.append(pix[1:21],np.append(wave,pix[-21:-1]))
				high_points = np.append(star_flux_chunk[1:21],np.append(high_points,star_flux_chunk[-21:-1]))

		#outlier rejection

				coef = np.polynomial.legendre.legfit(wave,high_points,5)
				fit = np.polynomial.legendre.legval(wave,coef)

				#fit = interpolate.UnivariateSpline(wave,high_points,k=2)

				for k in range(1):
					#condition1 = star_lam_chunk == wave
					condition2 = abs((high_points/fit)-1.0) < 0.2
					high_points = high_points[condition2]
					wave = wave[condition2]

					if (np.shape(wave)[0] < 10):
						break
					else:
						coef = np.polynomial.legendre.legfit(wave,high_points,5)
						fit = np.polynomial.legendre.legval(wave,coef)
						#fit = interpolate.UnivariateSpline(wave,high_points,k=1)

					print "outlier iteration"

		#strap down at ends

				wave = np.append(pix[1:21],np.append(wave,pix[-21:-1]))
				high_points = np.append(star_flux_chunk[1:21],np.append(high_points,star_flux_chunk[-21:-1]))

				coef = np.polynomial.legendre.legfit(wave,high_points,5)
				fit = np.polynomial.legendre.legval(wave,coef)

				full_fit = np.polynomial.legendre.legval(pix,coef)


				#plt.clf()
				#plt.plot(pix,star_flux_chunk, '--g')
				#plt.plot(wave,high_points, 'ro')
				#plt.plot(pix,full_fit, '--b')
				#plt.show()

				star_flux_chunk = star_data[j,0:]/full_fit

				norm_flux = np.append(norm_flux,np.fliplr([star_flux_chunk])[0])

				norm_lamb = np.append(norm_lamb,pix)


				#norm_flux = np.append(norm_flux,star_flux_chunk/full_fit)
				#norm_lamb = np.append(norm_lamb,pix)
		
				#plt.draw()
       				#plt.pause(1)

		#clean obvious

			#print norm_flux

			cond = (norm_flux < 2.5) & (norm_flux > 0.0)
			norm_lamb = norm_lamb[cond]
			norm_flux = norm_flux[cond]


		#smooth entire spectrum to average overlap

			star_norm_lamb,star_norm_flux = zip(*sorted(zip(norm_lamb,norm_flux)))
			star_norm_lamb=np.array(star_norm_lamb)
			star_norm_flux=np.array(star_norm_flux)	
			#has to be 2 so that overlap regions given equal weight between 2 orders
			star_norm_flux = np.convolve(star_norm_flux, np.ones((3,))/3)

			star_norm_flux = star_norm_flux[1:-1]

		#clip spikes

			m_norm_flux = np.convolve(star_norm_flux, np.ones((5,))/5)

			sz1 = np.shape(m_norm_flux)[0]
			sz2 = np.shape(star_norm_flux)[0]
			bound = np.int(np.abs(sz1-sz2)/2)

			m_norm_flux = m_norm_flux[bound:-bound]

			cond = np.abs(m_norm_flux - star_norm_flux) < 0.01

			star_norm_lamb = star_norm_lamb[cond]
			star_norm_flux = star_norm_flux[cond]


	#2D smoothing

			norm_lamb=np.array([])
			norm_flux=np.array([])

			padwidth = 200
			star_data = np.pad(star_data,padwidth,'edge')

			#median extrema outliers
			#ids = np.where((star_data > 1e6) | (star_data < 1e3))
			
			#star_data[ids] = star_data_norm_2[ids]

			#star_data_norm_2 = filt.median_filter(star_data,size=(2,2),mode="nearest")	
			star_data_norm_2 = filt.maximum_filter(star_data,size=(2,2),mode="nearest")
			#star_data_norm_2 = filt.maximum_filter(star_data_norm_2,size=(2,1),mode="nearest")
			star_data_norm_2 = filt.gaussian_filter(star_data_norm_2,sigma=(5,5),mode="nearest",truncate=2.0)
			
			#star_data_norm = np.fft.fftshift(np.fft.fft2(star_data))

			def butter_2d(x,y,c_x,c_y,D,n):
				x = np.arange(0,x)
				y = np.arange(0,y)
				u,v = np.meshgrid(x,y)
				return 1/( 1+ ( ( ((u-c_x)**2+(v-c_y)**2) /(D**2) ) )**n )

			#p = (10,100,0.25)
			#errorfunction = lambda p: np.ravel(butter_2d(*p)(gridx,gridy) - np.transpose(but_img))
			#fit = optimize.least_squares(errorfunction, p, bounds = bounds)
			#p = fit.x

			#but_img = butter_2d(2045,62,1023,31,7,0.5)

			#add padding for edge effects	
			#but_img = np.pad(but_img,padwidth,'edge')

			#star_data_norm = np.abs( np.fft.ifft2( np.fft.ifftshift(star_data_norm*but_img) ) )

			#f = plt.figure()
			#ax1=f.add_subplot(311)
			#plt.imshow(star_data,interpolation="none")
			#ax2=f.add_subplot(312)
			#plt.imshow(star_data_norm_2,interpolation="none")
			#ax2=f.add_subplot(313)
			#plt.imshow(but_img,interpolation="none")
			#plt.show()

			#sys.exit()

			star_data_norm_2 = star_data/star_data_norm_2

			#remove padding

			star_data_norm_2 = star_data_norm_2[padwidth:-padwidth,padwidth:-padwidth]

			for j in x:
				#reversing array

				j=np.int(j)
				norm_flux = np.append(norm_flux,np.fliplr([star_data_norm_2[j,0:]])[0])
				#norm_flux = np.append(norm_flux,star_data_norm_2[j,0:])
				
				pix = ((cal_array[np.int(j),1])*pix_s)+cal_array[np.int(j),2]
				norm_lamb = np.append(norm_lamb,pix)
					


		#clean obvious

			#print norm_flux

			cond = (norm_flux < 2.5) & (norm_flux > 0.0)
			norm_lamb = norm_lamb[cond]
			norm_flux = norm_flux[cond]

		#clip spikes

			m_norm_flux = np.convolve(norm_flux, np.ones((5,))/5)

			sz1 = np.shape(m_norm_flux)[0]
			sz2 = np.shape(norm_flux)[0]
			bound = np.int(np.abs(sz1-sz2)/2)

			m_norm_flux = m_norm_flux[bound:-bound]

			cond = np.abs(m_norm_flux - norm_flux) < 0.01

			star_norm_lamb_2 = norm_lamb[cond]
			star_norm_flux_2 = norm_flux[cond]

		#smooth entire spectrum to average overlap

			star_norm_lamb_2,star_norm_flux_2 = zip(*sorted(zip(star_norm_lamb_2,star_norm_flux_2)))
			star_norm_lamb_2=np.array(star_norm_lamb_2)
			star_norm_flux_2=np.array(star_norm_flux_2)	
			#has to be 2 so that overlap regions given equal weight between 2 orders
			star_norm_flux_2 = np.convolve(star_norm_flux_2, np.ones((2,))/2)

			star_norm_flux_2 = star_norm_flux_2[0:-1]

			flux2 = medfilt(star_norm_flux_2,kernel_size=41)

			#sys.exit()

	#END Continnum normalization

		#radial velocity correction
			H_beta = 4861
			H_gamma = 4341
			H_delta = 4102
			H_epsilon = 3970
			H_alpha = 6562
			#H_zeta = 3889

			dw = 160
			minid = np.array([])
			rvset = np.array([])

			H_wave = [H_beta,H_gamma,H_delta,H_epsilon,H_alpha]

			toff = SRP.find_line(star_norm_lamb_2,flux2,H_wave[0],dw)
			rvset = np.append(rvset,(toff))

			for w in H_wave[1:]:
				toff2 = SRP.find_line(star_norm_lamb_2,flux2,w+(toff-H_beta),dw/2)
				rvset = np.append(rvset,(toff2))

			#if np.abs(rvset[0]) < np.abs(rvset[1]):
			#	H_wave = H_wave[1:]
			#	rvset = rvset[1:]

			print H_wave,rvset
	
			param = np.polyfit(rvset,H_wave,1)
			func = np.poly1d(param)

			resid = H_wave-(func(rvset))
			print resid

			if (np.abs(np.sum(resid)) > 10):
				print "RV correction FUBAR"
				#sys.exit()

			rvcorr_lamb = func(star_norm_lamb)
			rvcorr_lamb_balmer = func(star_norm_lamb_2)

			rvset2 = np.array([])
			rvlinelist = np.sort(np.append(H_wave,[4045.88,4481.3,5055.984,5316.615,5534.847,5895.924,6158.187]))
			
			toff_last = 0.0
			for w in rvlinelist:
				toff2 = SRP.find_line(rvcorr_lamb,star_norm_flux,w-toff_last,6)
				rvset2 = np.append(rvset2,(toff2))
				toff_last = np.float(w-toff2)


			#non-linear wavelength correction

			star_dir = bal_dir+star.replace('*','')+'/'
			if not os.path.exists(star_dir):
				os.makedirs(star_dir)

			rv_resid = rvset2-rvlinelist

			param2 = np.polyfit(rvlinelist,rv_resid,2)
			func2 = np.poly1d(param2)

			plt.clf()
			plt.plot(rvcorr_lamb,func2(rvcorr_lamb),'g--')
			plt.plot(rvlinelist,rv_resid,'bo')
			plt.ylim([-8,8])
			plt.savefig(star_dir+'rv_nonlinear_corr_1.pdf')

			rvcorr_lamb2 = rvcorr_lamb-func2(rvcorr_lamb)
			rvcorr_lamb_balmer2 = rvcorr_lamb_balmer-func2(rvcorr_lamb_balmer)

			#iterate again

			rvset2 = np.array([])
			rvlinelist = np.sort(np.append(rvlinelist,[6371.37,6757.2,7771.73]))
			
			toff_last = 0.0
                        for w in rvlinelist:
                                toff2 = SRP.find_line(rvcorr_lamb2,star_norm_flux,w-toff_last,6)
                                rvset2 = np.append(rvset2,(toff2))
                                toff_last = np.float(w-toff2)


                        rv_resid = rvset2-rvlinelist

                        param2 = np.polyfit(rvlinelist,rv_resid,2)
                        func2 = np.poly1d(param2)

			plt.clf()
                        plt.plot(rvcorr_lamb2,func2(rvcorr_lamb2),'g--')
                        plt.plot(rvlinelist,rv_resid,'bo')
                        plt.ylim([-8,8])
                        plt.savefig(star_dir+'rv_nonlinear_corr_2.pdf')

                        rvcorr_lamb2 = rvcorr_lamb2-func2(rvcorr_lamb2)
                        rvcorr_lamb_balmer2 = rvcorr_lamb_balmer2-func2(rvcorr_lamb_balmer2)

			#plt.clf()
			#plt.plot(rvcorr_lamb+func2(star_norm_lamb),star_norm_flux,'b')
			#plt.plot(rvcorr_lamb2,star_norm_flux,'g')
			#plt.plot(rvcorr_lamb,star_norm_flux,'r')

			#for l in rvlinelist:
			#	plt.axvline(l,0,1.5,'k')

			#plt.ylabel("Normalized Intensity")
			#plt.xlabel("Wavelength (Angs)")
			#plt.show()
	
			#sys.exit()

			#plt.clf()
			#plt.plot(rvcorr_lamb_balmer,flux_balmer)
			#plt.ylim((0.6,1.2))
			#plt.show()


		#plot check final spectrum

			#plt.clf()
			#plt.plot(rvcorr_lamb,star_norm_flux,'b')
			#plt.plot(rvcorr_lamb_balmer,star_norm_flux_2,'r')
			#plt.ylim((0.6,1.2))
			#plt.show()

	#save data to hardrive

			hdu1 = pyfits.PrimaryHDU(zip(rvcorr_lamb,star_norm_flux))
			hdu2 = pyfits.PrimaryHDU(zip(rvcorr_lamb_balmer,star_norm_flux_2))
			hdulist1 = pyfits.HDUList([hdu1])
			hdr1 = hdulist1[0].header
			hdulist2 = pyfits.HDUList([hdu2])
			hdr2 = hdulist2[0].header

			try:

				hdr1['OBJECT'] = star_hdr["OBJECT"]
				hdr2['OBJECT'] = star_hdr["OBJECT"]

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

				hdr1['OBJECT'] = name
				hdr2['OBJECT'] = name
							 
			try:
				hdr1['INSTRUME'] = star_hdr["INSTRUME"]
				hdr2['INSTRUME'] = star_hdr["INSTRUME"]
			except (KeyError):
				hdr1['INSTRUME'] = "unknown"
				hdr2['INSTRUME'] = "unknown"

			try:
				hdr1['OBSDATE'] = star_hdr["OBSDATE"]
				hdr2['OBSDATE'] = star_hdr["OBSDATE"]
			except (KeyError):
				hdr1['DATE-OBS'] = star_hdr["DATE-OBS"]
				hdr2['DATE-OBS'] = star_hdr["DATE-OBS"]

			hdulist1[0].header = hdr1
			hdulist2[0].header = hdr2

			str_flux_norm_leg = str_star.replace("[0]","").replace(".fits","_cntnorm_legendre.fits").replace(".ec","")
			str_flux_norm_twod = str_star.replace("[0]","").replace(".fits","_cntnorm_twod.fits").replace(".ec","")

			hdulist1.writeto(cntnorm_dir+str_flux_norm_leg,clobber=True)	
			hdulist2.writeto(cntnorm_dir+str_flux_norm_twod,clobber=True)

	#remove temporary files to save hardrive space

			if(beast):
				temp_files_del = os.listdir(temp_dir)
				for f in temp_files_del:
					os.remove(temp_dir+f)

		elif proc in "n":
			print "Skipping step"
		elif proc in "b":
			break
		else:
			print "Bugged input"
	
		end = time.time()

		print "elapsed time for star: ",np.int(end-start)," sec"


#clean up
	#os.remove('master_dark.fits')

	

		


			
