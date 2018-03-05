#! /usr/bin/env python

from __future__ import unicode_literals

import sys
import os
import pyfits
import numpy as np
import subprocess
import threading
import shlex
import pickle
import matplotlib
matplotlib.use('Agg')
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
import glob
import shutil
import csv

def elementnumber(str_in):
	Z=0.0

	if 'He' in str_in:
		Z=2.0
	elif 'H' in str_in:
		Z=1.0
	elif 'Li' in str_in:
		Z=3.0
	elif 'Be' in str_in:
		Z=4.0
	elif 'B' in str_in:
		Z=5.0
	elif 'Cr' in str_in:
		Z=24.0
	elif 'Ca' in str_in:
		Z=20.0
	elif 'C' in str_in:
		Z=6.0
	elif 'O' in str_in:
		Z=8.0
	elif 'Fe' in str_in:
		Z=26.0
	elif 'F' in str_in:
		Z=9.0
	elif 'Ne' in str_in:
		Z=10.0
	elif 'Na' in str_in:
		Z=11.0
	elif 'Ni' in str_in:
		Z=28.0
	elif 'N' in str_in:
		Z=7.0
	elif 'Mg' in str_in:
		Z=12.0
	elif 'Mn' in str_in:
		Z=25.0
	elif 'Zr' in str_in:
		Z=40.0
	elif 'Ti' in str_in:
		Z=22.0
	elif 'V' in str_in:
		Z=23.0
	elif 'Si' in str_in:
		Z=14.0
	elif 'S' in str_in:
		Z=16.0
	elif 'Y' in str_in:
		Z=39.0
	elif 'Al' in str_in:
		Z=13.0
	else:
		Z=9999.0

	if '1' in str_in:
		Z=Z+0.0
	elif '2' in str_in:
		Z=Z+0.1
	elif '3' in str_in:
		Z=Z+0.2
	elif '4' in str_in:
		Z=Z+0.3
	else:
		Z=Z+0.9999

	return str(Z)


def writetomoog(filename,linelist):

	out = open(filename+".moog",'w')

	out.write('# ILS Linelist from McD_pipe\n')

	for line in linelist:
		elnum = line[0]
		eV = line[2]
		loggf = line[3]
		if loggf < 0: 
			char = '{:<8}'
			char2 = '{:<32}'
		else:
			char = '{:<9}'
			char2 = '{:<31}'
	
		line2 = ''.join(('{:<14}'.format(line[1]),'{:<10}'.format(elnum),char.format(str(eV)),char2.format(str(loggf)),'{:<8}'.format(line[4])))
		out.write(line2+'\n')
	
	out.close

	return filename+".moog"


def microturbulence_perscription(teff):
	#Gebran, Monier, Royer, Lobel, Blomme, conf. pro 2013 arxiv 1312.0442
	import numpy as np
    
	vturb = 3.31*np.exp(-(np.log(teff/8071.03)**2/0.01045))
    
	return vturb

## pyBALMER9 ##

def execute_balmer9(datfile):
	output = 'FortranOutputUnit7_Balmer9.dat'
	try:
		os.remove(output)
	except(IOError,OSError):
		pass
	os.system('cp '+datfile+' ./auto.dat')
	#os.system('source /isluga3/kim/intel/bin/ifortvars.csh')
	os.system('sh auto_balmer9.com > auto.log')

def execute_width9(datfile):
	#output = 'FortranOutputUnit7_Balmer9.dat'
	#os.remove(output)
	#os.system('cp '+datfile+' ./auto.dat')
	#os.system('source /isluga3/kim/intel/bin/ifortvars.csh')
	os.system('sh auto_width9.com > auto.log')

def execute_moog(folder,star):
	import numpy as np
	from SRP import read_moog_abund
	import shlex
	import subprocess
	import threading

	cmd = '~/.local/bin/MOOGSILENT > moog.log'
	cmd = shlex.split(cmd)
	#print cmd
	proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
	timer = threading.Timer(25,proc.kill)
	timer.start()
	proc.communicate()
	if timer.is_alive():
		timer.cancel()
		shutil.copyfile('out2',folder+'/'+star+'_moog_output.out2')
		return read_moog_abund('out2')
	else:
		return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
	#raise SubprocessTimeoutError('MOOG Failed')

def execute_moog_synth():
	import numpy as np
	from SRP import read_moog_abund
	import shlex
	import subprocess
	import threading
	

	cmd = '~/.local/bin/MOOGSILENT > moog.log'
	cmd = shlex.split(cmd)
	#print cmd
	proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
	timer = threading.Timer(25,proc.kill)
	timer.start()
	proc.communicate()
	if timer.is_alive():
		timer.cancel()
		return 1
	else:
		return 0
	#raise SubprocessTimeoutError('MOOG Failed')

def create_moog_file(model_in,lines_in):
	"""Creates the MOOG driver file."""
	file_name = "batch.par"
	f = open(file_name, 'w')
	f.write('abfind\n')
	f.write('standard_out out1\n')
	f.write('summary_out  out2\n')
	f.write("model_in     '"+model_in+"'\n")
	f.write("lines_in     '"+lines_in+"'\n")
	f.write('atmosphere   2\n')
	f.write('molecules    2\n')
	f.write('plot         2\n')
	f.write('lines        2\n')
	f.write('strong       0\n')
	f.write('flux/int     0\n')
	f.write('damping      2\n')
	f.write('trudamp      1\n')
	f.write('strong       0\n')
	f.close()

def read_moog_abund(filename):
	import numpy as np

	f = open(filename,'r')
	lines = f.readlines()
	f.close()

	a = []
	key = ["T_eff","logg"]
	for l in lines:
		if 'T=' in l:
			t=l.split(" ")
			t = filter(None, t)
			teff = t[0].split('=')
			logg = t[1].split('=')
			#print t
			a=np.append([a],[np.float(teff[1]),np.float(logg[1])])

		elif 'Abundance Results for Species' in l:
			t=l.split(" ")
			t=filter(None,t)
			key=np.append([key],[t[4]+t[5]])

		elif 'average abundance =' in l:
			t=l.split(" ")
			t=filter(None,t)
			#print t
			if '*' in t[3]:
				a=np.append([a],[np.nan])
			else:
				a=np.append([a],[np.float(t[3])])
				
	return key,a

#################### BALMER9 INTERFACE ####################

def read_data(filename,skip):
	import numpy as np

	f = open(filename,'r')
	lines = f.readlines()
	f.close()

	data = lines[skip:]

	arr = 0
	for d in data:
		a = d.split(' ')
		a = filter(None, a)
		for i,x in enumerate(a): a[i] = x.replace('\n','')
		a = filter(None, a)
		if arr == 0: arr = [a]
		else: arr = np.concatenate((arr,[a]),axis=0)

	return arr


def col_pair(arr,col1,col2):
	import numpy as np

	c1 = np.array(arr[0:,col1])
	c1 = c1.astype(np.float)
	c2 = np.array(arr[0:,col2])
	c2 = c2.astype(np.float)
	return np.append(c1,c2)

def find_line(lamb,flux,lw,dw):
	import numpy as np

	minid=np.array([])
	ids = np.where((lamb < lw+dw) & (lamb > lw-dw))
	tlamb = np.array(lamb[ids])
	tflux = np.array(flux[ids])
	for l in trough_min(tflux): minid = np.append(minid,tlamb[np.where(l == tflux)])

	#plt.clf()
	#plt.plot(tlamb,tflux)
	#plt.axvline(np.median(minid),0,2,'k--')
	#plt.show()

	return np.median(minid)

def chisqr_balmer9(data):
	import numpy as np
	from SRP import find_line
	from SRP import col_pair
	from SRP import read_data


	output = './FortranOutputUnit7_Balmer9.dat'

	hbetaline = np.float(4861.230)
	hgammaline = np.float(4340.47)
	hdeltaline = np.float(4101.74)

	data_lam = data[0]
	data_lam = data_lam.astype(np.float)
	data_flx = data[1]
	data_flx = data_flx.astype(np.float)

	b9_out = read_data(output,5)

	angs = col_pair(b9_out,1,2)
	ids = np.argsort(angs)
	angs = angs[ids]

#H Beta

	hbeta = col_pair(b9_out,5,6)
	hbeta = hbeta[ids]
	angsb = angs + hbetaline
	f_beta = interp1d(angsb, hbeta, fill_value="extrapolate")
	cond = (data_lam > min(angsb)) & (data_lam < max(angsb)) 

	data_flx_beta = data_flx[cond]
	data_lam_beta = data_lam[cond]
		
	#correct broad continuum if still present

	sz = np.size(data_flx_beta)
	subsel = np.floor(sz*0.1)
	ind = np.argpartition(data_flx_beta, -subsel)[-subsel:]
	foff = np.median(data_flx_beta[ind])
	data_flx_beta = data_flx_beta+(1-foff)

	#correct wavelength offsets still present

	toff = find_line(data_lam_beta,data_flx_beta,hbetaline,30)
	data_lam_beta = data_lam_beta - (toff-hbetaline)

	angsb = angs + hbetaline
	#plt.clf()
	#plt.title("H Beta")
	#plt.plot(angsd,hbeta,'r-')
	#plt.plot(data_lam_beta,data_flx_beta,'b')
	#plt.xlim([4800,4920])
	#plt.draw()

	cond = (data_lam_beta > min(angsb)) & (data_lam_beta < max(angsb))
	chisqr = np.square(f_beta(data_lam_beta) - data_flx_beta) / data_flx_beta 
	dlam = np.abs(data_lam_beta-hbetaline)

	tab_p_prime = data_flx_beta-np.roll((data_flx_beta),4)
	tab_m_prime = np.roll((data_flx_beta),-4)-data_flx_beta
	tab_p_prime2 = data_flx_beta-np.roll((data_flx_beta),7)
	tab_m_prime2 = np.roll((data_flx_beta),-7)-data_flx_beta
	condition = (tab_p_prime > 0) & (tab_m_prime < 0) & (tab_p_prime2 > 0) & (tab_m_prime2 < 0) | ((dlam > 1) & (dlam < 10))

	chisqr = np.sum(chisqr[condition])

	beta_line = [data_lam_beta,data_flx_beta,f_beta(data_lam_beta),condition]

#H Gamma

	hgamma = col_pair(b9_out,7,8)
	hgamma = hgamma[ids]
	angsg = angs + hgammaline
	f_gamma = interp1d(angsg, hgamma, fill_value="extrapolate")
	cond = (data_lam > min(angsg)) & (data_lam < max(angsg)) 

	data_flx_gamma = data_flx[cond]
	data_lam_gamma = data_lam[cond]

	#correct broad continuum if still present

	sz = np.size(data_flx_gamma)
	subsel = np.floor(sz*0.1)
	ind = np.argpartition(data_flx_gamma, -subsel)[-subsel:]
	foff = np.median(data_flx_gamma[ind])
	data_flx_gamma = data_flx_gamma+(1-foff)

	#correct wavelength offsets still present

	toff = find_line(data_lam_gamma,data_flx_gamma,hgammaline,30)
	data_lam_gamma = data_lam_gamma - (toff-hgammaline)

	#plt.clf()
	#plt.title("H Gamma")
	#plt.plot(angsd,hgamma,'r-')
	#plt.plot(data_lam_gamma,data_flx_gamma,'b')
	#plt.xlim([4280,4400])
	#plt.draw()

	cond = (data_lam_gamma > min(angsg)) & (data_lam_gamma < max(angsg))
	chisqr2 = np.square(f_gamma(data_lam_gamma) - data_flx_gamma) / data_flx_gamma 
	dlam = np.abs(data_lam_gamma-hgammaline)

	tab_p_prime = data_flx_gamma-np.roll((data_flx_gamma),4)
	tab_m_prime = np.roll((data_flx_gamma),-4)-data_flx_gamma
	tab_p_prime2 = data_flx_gamma-np.roll((data_flx_gamma),7)
	tab_m_prime2 = np.roll((data_flx_gamma),-7)-data_flx_gamma
	condition = (tab_p_prime > 0) & (tab_m_prime < 0) & (tab_p_prime2 > 0) & (tab_m_prime2 < 0) | ((dlam > 1) & (dlam < 10))

	chisqr2 = np.sum(chisqr2[condition])

	gamma_line = [data_lam_gamma,data_flx_gamma,f_gamma(data_lam_gamma),condition]

#H Delta

	hdelta = col_pair(b9_out,9,10)
	hdelta = hdelta[ids]
	angsd = angs + hdeltaline
	f_delta = interp1d(angsd, hdelta, fill_value="extrapolate")
	cond = (data_lam > min(angsd)) & (data_lam < max(angsd)) 

	data_flx_delta = data_flx[cond]
	data_lam_delta = data_lam[cond]

	#correct broad continuum if still present

	sz = np.size(data_flx_delta)
	subsel = np.floor(sz*0.1)
	ind = np.argpartition(data_flx_delta, -subsel)[-subsel:]
	foff = np.median(data_flx_delta[ind])
	data_flx_delta = data_flx_delta+(1-foff)

	#correct wavelength offsets still present

	toff = find_line(data_lam_delta,data_flx_delta,hdeltaline,30)
	data_lam_delta = data_lam_delta - (toff-hdeltaline)

	#print toff

	#print np.shape(data_lam_beta)
	#print np.shape(data_flx_beta)
	#print np.shape(angs)
	#print np.shape(hdelta)

	#plt.clf()
	#plt.title("H Delta")
	#plt.plot(angsd,hdelta,'r-')
	#plt.plot(data_lam_delta, data_flx_delta,'b')
	#plt.xlim([4040,4160])
	#plt.draw()

	cond = (data_lam_delta > min(angsd)) & (data_lam_delta < max(angsd))
	chisqr3 = np.square(f_delta(data_lam_delta) - data_flx_delta) / data_flx_delta 
	dlam = np.abs(data_lam_delta-hdeltaline)

	tab_p_prime = data_flx_delta-np.roll((data_flx_delta),4)
	tab_m_prime = np.roll((data_flx_delta),-4)-data_flx_delta
	tab_p_prime2 = data_flx_delta-np.roll((data_flx_delta),7)
	tab_m_prime2 = np.roll((data_flx_delta),-7)-data_flx_delta
	condition = (tab_p_prime > 0) & (tab_m_prime < 0) & (tab_p_prime2 > 0) & (tab_m_prime2 < 0) | ((dlam > 1) & (dlam < 10))

	chisqr3 = np.sum(chisqr3[condition])

	delta_line = [data_lam_delta,data_flx_delta,f_delta(data_lam_delta),condition]
	

	#plot_balmer_lines([shft_lam,data_flx])
	print chisqr,chisqr2,chisqr3
	chisqr_final = np.array([chisqr,chisqr2,chisqr3])

	return chisqr_final,beta_line,gamma_line,delta_line


def plot_balmer_lines(data):
	import numpy as np

	output = './FortranOutputUnit7_Balmer9.dat'

	arr = read_data(output,5)
		
	angs = col_pair(arr,1,2)
	ids = np.argsort(angs)
	angs = angs[ids]

	halpha = col_pair(arr,3,4)
	halpha = halpha[ids]

	hbeta = col_pair(arr,5,6)
	hbeta = hbeta[ids]

	hgamma = col_pair(arr,7,8)
	hgamma = hgamma[ids]

	hdelta = col_pair(arr,9,10)
	hdelta = hdelta[ids]

	data_lam = data[0]
	data_flx = data[1]

	hbetaline = 4861.230
	hgammaline = 4340.47
	hdeltaline = 4101.74

	ax = plt.subplot(1,1,1)

	angsb = angs + hbetaline
	ax.plot(angsb,hbeta,'r-')

	angsg = angs + hgammaline
	ax.plot(angsg,hgamma,'r-')

	angsd = angs + hdeltaline
	ax.plot(angsd,hdelta,'r-')
	ax.plot(data_lam,data_flx,'b')
	ax.set_xlim(4800,4920)

	plt.show()

#def pybalmer9(linelist,data,star):

#################### END BALMER9 INTERFACE ####################

def outputnames(files,tag):

	filesb = list(files)
	for idx,i in enumerate(filesb):
		j = str(i)
		filesb[idx]=j.replace(".fits","_"+tag+".fits")
	return filesb

def gauss_function(x, a, m, sigma, x0):
	import numpy as np
    	return a*np.exp(-(x-m)**2/(2*sigma**2))+x0

def gauss_function_2(x, a, m, sigma, x0, a_2, m_2):
	import numpy as np
    	return a*np.exp(-(x-m)**2/(2*sigma**2))+(a*a_2)*np.exp(-(x-(m+m_2))**2/(2*sigma**2))+x0

def gauss_function_3(x, a, m, sigma, x0, m_2, m_3, a_1, a_2, a_3):
	import numpy as np
    	return (a*a_1)*np.exp(-(x-m)**2/(2*sigma**2)) + (a*a_2)*np.exp(-(x-(m+m_2))**2/(2*sigma**2)) + (a*a_3)*np.exp(-(x-(m+m_3))**2/(2*sigma**2)) + x0

def gaussian(height, center_x, center_y, width_x, width_y):
	import numpy as np
	"""Returns a gaussian function with the given parameters"""
	width_x = float(width_x)
	width_y = float(width_y)
	return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def gauss_function_2d((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
	import numpy as np
    	xo = float(xo)
    	yo = float(yo)
    	a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    	g = offset + amplitude*np.exp(-(a*((x-xo)**2)+2*b*(x-xo)*(y-yo)+c*((y-yo)**2)))
    	return g.ravel()

def trough_min(numbers):
    m1, m2, m3, m4, m5 = float('inf'), float('inf'), float('inf'), float('inf'), float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2, m3, m4, m5 = x, m1, m2, m3, m4
        elif x < m2:
            m2 = x
	elif x < m3:
            m3 = x
	elif x < m4:
            m4 = x
	elif x < m5:
            m5 = x
    #print m1,m2,m3,m4,m5
    return [m1,m2,m3,m4,m5]

def get_mulList(*args):
        return map(list,zip(*args))


def strom_phot(star):
	from astroquery.vizier import Vizier
	import numpy as np
	import os

	EP2015_stromgren = Vizier(catalog="J/A+A/580/A23")

	result = EP2015_stromgren.query_object(star)

	#print result

	if not result:
		return 0

	#print result[0].keys()
	by = result[0]["b-y"][0]
	c1 = result[0]["c1"][0]
	m1 = result[0]["m1"][0]
	beta = result[0]["beta"][0]

	if not beta:
		return 0

	if (2.72 < beta < 2.88) & (0.05 < by < 0.22):
		n = 6
	elif (2.87 < beta < 2.93) & (-0.01 < by < 0.06):
		n = 5
	elif (2.60 < beta < 2.72) & (0.22 < by < 0.39):
		n = 7
	elif (0.02 < m1 < 0.76) & (0.39 < by < 1.00):
		n = 8
	else:
		n = 6

	with open("starParam.pro","wb") as idl_code:
		idl_code.write("uvbybeta,"+str(by)+','+str(m1)+','+str(c1)+','+str(beta)+','+str(n)+',Teff\n')
		idl_code.write("print,'readline: ',Teff")

	os.system("nohup nice $IDL_DIR/bin/idl < starParam.pro > idlout.txt")

	idl_output = open("idlout.txt","rb")

	out = idl_output.readlines()
	temp = np.float(out[-1][9:]) 

	return temp

def heat_map_fit(hmap,x,y,gridx,gridy):
	from scipy.optimize import least_squares
	from SRP import gaussian

	bounds = ([0,np.min(x),np.min(y),50,0.1],[2,np.max(x),np.max(y),1000,2.0])

	maxid = np.where(hmap == np.max(hmap))
	p = (np.max(hmap),x[maxid[0][0]],y[maxid[1][0]],500,1.0)
	errorfunction = lambda p: np.ravel(gaussian(*p)(gridx,gridy) - np.transpose(hmap))
	try:
		fit = least_squares(errorfunction, p, bounds = bounds)
		p = fit.x	
	except ValueError:
		pass

	return p

#################### MOOG SYNTH INTERFACE ####################

def write_linelist_moog(filename,linelist):
	'''
	Function:	create a linelist MOOG can read

	Variables:	filename = filename of the linelist to use
			linelist = array of line data [element number, wavelength, excitation potential, log oscillator strength, EW of line]

	Expectations:  	None
	'''

	from SRP import elementnumber

	out = open(filename,'w')

	out.write('# ILS Linelist from McD_pipe\n')

	for line in linelist:
		elnum = elementnumber(line[0])
		eV = line[2]
		loggf = line[3]
		if loggf < 0: 
			char = '{:<8}'
			char2 = '{:<32}'
		else:
			char = '{:<9}'
			char2 = '{:<31}'
	
		line2 = ''.join(('{:<14}'.format(line[1]),'{:<10}'.format(elnum),char.format(str(eV)),char2.format(str(loggf)),'{:<8}'.format(line[4])))
		out.write(line2+'\n')
	
	out.close

	return filename

def write_starxy(star,wave,flux):
	'''
	Function:	create a 'star.xy' file for MOOG to read

	Variables:	star = name of star to put at the top of the file
			wave = wavelengths
			flux = take a wild guess	

	Expectations:  	None
	'''
	#create observed star spectrum for moog synth 
	file_name = "star.xy"
	f = open(file_name, 'w')
	f.write(star+'\n')
	for x,y in zip(wave,flux):
		f.write(str("{:10.4f}".format(x))+' '+str("{:10.4f}".format(y))+'\n')
	
	f.close()

def create_moog_file_synth(model_in,wmin,wmax,wdisp,wop,fmin,fmax,elements,g_fwhm):
	'''
	Function:	create a moog synth parameter file using gaussian broadening

	Variables:	model_in = stellar model of choice compatable with MOOG
			wmin = bluest wavelength to synthesize
			wmax = reddest wavelength to synthesize
			wdisp = step size resolution of synthesis 
			wop = 1.0, just always 1.0, unless really desired normalization offset
			fmin = flux minimum for plotting
			fmax = flux maximum for plotting
			elements = array of tuples for element abundances
			vrot = rotational broadening velocity

	Expectations:  	The linelist to use is called 'regional_linelist.moog'
	'''
	import numpy as np
	import decimal

	threeplaces = decimal.Decimal(10) ** -3
	twoplaces = decimal.Decimal(10) ** -2
	wdisp = decimal.Decimal(str(wdisp)).quantize(threeplaces)
	g_fwhm = decimal.Decimal(str(g_fwhm)).quantize(threeplaces)

	wdisp = np.max([wdisp,0.001])

	"""Creates the MOOG driver file."""
	file_name = "batch.par"
	f = open(file_name, 'w')
	f.write('synth\n')
	f.write("terminal     'nodevice'\n")     
	f.write("standard_out 'out1'\n")
	f.write("summary_out  'out2'\n")
	f.write("smoothed_out 'out3'\n")
	f.write("model_in     '"+model_in+"'\n")
	f.write("lines_in     'regional_linelist.moog'\n")
	f.write("observed_in  'star.xy'\n")
	f.write('atmosphere   1\n')
	f.write('molecules    2\n')
	f.write('plot         1\n')
	f.write('lines        1\n')
	f.write('strong       0\n')
	f.write('flux/int     0\n')
	f.write('damping      1\n')
	f.write('strong       0\n')
	f.write('abundances   '+str(np.size(elements)/2)+'    1\n')
	for e,a in elements:
		#print e,a  
		a = decimal.Decimal(str(a)).quantize(twoplaces)
		f.write('     '+str(np.int(e))+'     '+str(a)+'\n')
	f.write('synlimits\n')
	f.write('  '+str(np.int(wmin))+'  '+str(np.int(wmax))+'  '+str(wdisp)+'  '+str(wop)+'\n')
	#f.write('obspectrum   5\n')
	f.write('plotpars   1\n')
	f.write('  '+str(np.int(wmin))+'  '+str(np.int(wmax))+'  '+str(fmin)+'  '+str(fmax)+'\n')
	f.write('  0.0   0.0   0.0   0.0\n')
	f.write('  g   '+str(g_fwhm)+'   0.0   0.0   0.0   0.0\n')
	
	f.close()

def create_moog_file_synth_rot(model_in,wmin,wmax,wdisp,wop,fmin,fmax,elements,vrot):
	'''
	Function:	create a moog synth parameter file using rotational velocity broadening

	Variables:	model_in = stellar model of choice compatable with MOOG
			wmin = bluest wavelength to synthesize
			wmax = reddest wavelength to synthesize
			wdisp = step size resolution of synthesis 
			wop = 1.0, just always 1.0, unless really desired normalization offset
			fmin = flux minimum for plotting
			fmax = flux maximum for plotting
			elements = array of tuples for element abundances
			vrot = rotational broadening velocity

	Expectations:  	Naming the file batch.par to conform with other functions
	'''
	import numpy as np
	import decimal

	threeplaces = decimal.Decimal(10) ** -3
	twoplaces = decimal.Decimal(10) ** -2
	oneplace = decimal.Decimal(10) ** -1
	wdisp = decimal.Decimal(str(wdisp)).quantize(threeplaces)
	vrot = decimal.Decimal(str(vrot)).quantize(oneplace)

	wdisp = np.max([wdisp,0.001])
	print model_in
	"""Creates the MOOG driver file."""
	file_name = "batch.par"
	f = open(file_name, 'w')
	f.write('synth\n')
	f.write("terminal     'nodevice'\n")    
	f.write("standard_out 'out1'\n")
	f.write("summary_out  'out2'\n")
	f.write("smoothed_out 'out3'\n")
	f.write("model_in     '"+model_in+"'\n")
	f.write("lines_in     'regional_linelist.moog'\n")
	f.write("observed_in  'star.xy'\n")
	f.write('atmosphere   1\n')
	f.write('molecules    2\n')
	f.write('plot         1\n')
	f.write('lines        1\n')
	f.write('strong       0\n')
	f.write('flux/int     0\n')
	f.write('damping      1\n')
	f.write('strong       0\n')
	f.write('abundances   '+str(np.size(elements)/2)+'    1\n')
	for e,a in elements:
		#print e,a        
		a = decimal.Decimal(str(a)).quantize(twoplaces)
		f.write('     '+str(np.int(e))+'     '+str(a)+'\n')
	f.write('synlimits\n')
	f.write('  '+str(np.int(wmin))+'  '+str(np.int(wmax))+'  '+str(wdisp)+'  '+str(wop)+'\n')
	#f.write('obspectrum   5\n')
	f.write('plotpars   1\n')
	f.write('  '+str(np.int(wmin))+'  '+str(np.int(wmax))+'  '+str(fmin)+'  '+str(fmax)+'\n')
	f.write('  0.0   0.0   0.0   0.0\n')
	f.write('  r   0.0   '+str(vrot)+'   0.0   '+str(0.0)+'   0.0\n')
	
	f.close()

			
def read_starxy(star):
	'''
	Function:	reads star.xy formated MOOG data file

	Variables:	star = 'star.xy' file for observational data

	Expectations:  	None
	'''

	import numpy as np
	f = open(star,'r')

	flux_d = np.array([])
	wave_d = np.array([])
	for num,line in enumerate(f):
		if (num > 0):
			l = filter(None,line.split(" "))
			try:
				l = np.array(l).astype(np.float)
			except ValueError:
				continue
			flux_d = np.append(flux_d,l[1])
			wave_d = np.append(wave_d,l[0])
			#print l

	f.close()
	return wave_d,flux_d

def read_moog_out2():
	f = open('out2','r')

	data_line = False
	wave_line = False
	flux = np.array([])
	for line in f:
		abund_dict = {}

		if (data_line):
			l = line.split(" ")
			l.remove('')
			l = np.array(l).astype(np.float)
			#print l
			flux = np.append(flux,l)
			continue
		if (wave_line):
			w_start = np.float(line[:13])
			w_end = np.float(line[14:27])
			w_diff = np.float(line[28:33])
			norm = np.float(line[34:])
			#print w_start,w_end,w_diff,norm
			data_line = True
			continue
		if ("ALL abundances" in line):
			continue
		if ("element" in line):
			elstr = line[7:10]
			elstr.replace(" ","")
			abstr = line[24:]
			abstr.replace(" ","")

			abund_dict[elstr] = np.float(abstr)

			continue        
		if ("MODEL" in line):
			wave_line = True
			continue

	#print flux

	wave = np.arange((w_end-w_start)/w_diff+1)*w_diff+w_start

	f.close()
	return wave,flux

def read_out3(out):
	'''
	Function:	reads an out3 file

	Variables:	out = 'out3' file from moog

	Expectations:  	None
	'''

	import numpy as np
	f = open(out,'r')

	flux_m = np.array([])
	wave_m = np.array([])
	for num,line in enumerate(f):
		if (num > 1):
		    l = filter(None,line.split(" "))
		    l = np.array(l).astype(np.float)
		    flux_m = np.append(flux_m,l[1])
		    wave_m = np.append(wave_m,l[0])
		    #print l

	f.close()
	return wave_m,flux_m

def xcorr_moog_synth(star,out,dw_r):
	'''
	Function:	compute wavelength offset between model and data using cross correlation (dw)

	Variables:	out = 'out3' file from moog
			star = 'star.xy' file for observational data
			dw_r = search range of cross correlation

	Expectations:  	None
	'''

	import numpy as np
	from scipy.interpolate import interp1d
	from SRP import read_starxy
	from SRP import read_out3
	import matplotlib.pyplot as plt

	wave_d,flux_d = read_starxy(star)
	wave_m,flux_m = read_out3(out)

	model = interp1d(wave_m,flux_m)
	trim_id = (wave_d > np.min(wave_m)) & (wave_d < np.max(wave_m))
	wave_d = wave_d[trim_id]
	flux_d = flux_d[trim_id]

	flux_m = model(wave_d)

	flux_d = np.abs(flux_d-1)
	flux_m = np.abs(flux_m-1)

	x = np.abs(wave_d-np.roll(wave_d,1))
	#del_w = np.median([v for v in x if str(v) != 'nan'])
	del_w = np.nanmedian(x) 
	#print del_w
    
	try:
		rng = np.int(np.floor(dw_r/del_w))
	except ValueError:
		#not ideal but cant figure something beter if a nan pops up    
		try:
			rng = np.int(np.floor(dw_r/(x[0]-x[1])))
		except IndexError:
			return 0.00            

	xcorr_arr = np.array([])
	steps = np.arange(dw_r*rng)-rng
	for s in steps:
		xcorr = np.sum(np.roll(flux_d,s)*flux_m)
		xcorr_arr = np.append(xcorr_arr,xcorr)

	#check xcorrelation
	#plt.plot(steps*del_w,xcorr_arr)
	#plt.show()
	#print np.max(xcorr_arr)

	id_max = np.where(xcorr_arr == np.max(xcorr_arr))[0]
	shift = (steps[id_max][0]*del_w)

	if (shift > dw_r):
		return 0.00
	else:
		return shift
   
def moog_synth_chisqr(out,star,dw):
	'''
	Function:	calculates the chi-square statistic of MOOG files

	Variables:	out = 'out3' file from moog
			star = 'star.xy' file for observational data
			dw = offset between model and data

	Expectations:   
	'''

	import numpy as np
	from scipy.interpolate import interp1d
	from SRP import read_starxy
	from SRP import read_out3

	wave_d,flux_d = read_starxy(star)

	wave_d = wave_d - dw

	wave_m,flux_m = read_out3(out)

	#interpolate model
	model = interp1d(wave_m,flux_m)

	#trim data to model dimensions
	trim_id = (wave_d > np.min(wave_m)) & (wave_d < np.max(wave_m))
	wave_d = wave_d[trim_id]
	flux_d = flux_d[trim_id]

	#print trim_id

	#plt.plot(wave_d,model(wave_d))
	#plt.plot(wave_d,flux_d)
	#plt.show()
	m = model(wave_d)
	ids = np.where(m < 1.0)
    
	m = m[ids]
	o = flux_d[ids]
	#w = np.size(m)
	chisqr = np.sum((m-o)**2)

	return chisqr

def moog_synth_chisqr_xy(out,star,dw):
	'''
	Function:	calculates the chi-square statistic of MOOG files

	Variables:	out = 'out3' file from moog
			star = 'star.xy' file for observational data
			dw = offset between model and data

	Expectations:   
	'''

	import numpy as np
	from scipy.interpolate import interp1d
	from SRP import read_starxy
	from SRP import read_out3

	wave_d,flux_d = read_starxy(star)

	wave_d = wave_d - dw

	wave_m,flux_m = read_out3(out)

	#interpolate model
	model = interp1d(wave_m,flux_m)

	#trim data to model dimensions
	trim_id = (wave_d > np.min(wave_m)) & (wave_d < np.max(wave_m))
	wave_d = wave_d[trim_id]
	flux_d = flux_d[trim_id]

	#print trim_id

	#plt.plot(wave_d,model(wave_d))
	#plt.plot(wave_d,flux_d)
	#plt.show()
	m = model(wave_d)
	ids = np.where(m < 1.0)
    
	m = m[ids]
	o = flux_d[ids]
	w = wave_d[ids]
	#w = np.size(m)
	chisqr_y = np.sum((m-o)**2)

	m1,w1 = zip(*sorted(zip(m,w)))
	o1,w2 = zip(*sorted(zip(o,w)))
	dw = (np.array(w1)-np.array(w2))
	#print dw
	chisqr_x = np.sum((dw)**2)

	return chisqr_y,chisqr_x


def pymoogsilent():
	'''
	Function:	Execute moog silent through creating a new shell, required work around since os.system execution of MOOGSILENT doesnt work with synth

	Variables:	None

	Expectations:  MOOG executable location is defined and parameter files is called batch.par
	'''	
	#from future import unicode_literals

	import pexpect
	import sys
	import time

	# Don't do this unless you like being John Malkovich
	# c = pexpect.spawnu('/usr/bin/env python ./python.py')

	# Note that, for Python 3 compatibility reasons, we are using spawnu and
	# importing unicode_literals (above). spawnu accepts Unicode input and
	# unicode_literals makes all string literals in this script Unicode by default.
	try:
		c = pexpect.spawnu('/usr/local/moog/MOOG')
		c.expect(u'filename')
		c.sendline(u'batch.par')
		#sys.stdout.write(c.after)
		c.expect(u'choice')
		c.sendline(u'q')
		#sys.stdout.write(c.after)
		c.expect(pexpect.EOF)
		#sys.stdout.write(c.after)
		ct = time.time()
		nt = time.time()
		while(c.isalive() or (nt-ct > 30)):
			c.sendline(u'q')
			nt = time.time()

		if (nt-ct > 30):
			print 'MOOG stalled'
		#else:
			#print 'buda, but buda, but budda'
		return False
	except:
		print 'Fuck you I wont do what you tell me!'
		return True

def pysynth(x0,id_el,elements,dw,g_fwhm,selected_model,region,dwin,b_win):
	'''
	Function:	Wrtie a parameter file for synth module called batch.par

	Variables:	x0 = abundance values
			id_el = element numbers which are allowed to vary
			elements = full array of tuples, [(element number,abundance)]
			dw = wavelength offset between data and model
			g_fwhm = convolve model with gaussian
			selected_model = MOOG capable stelalr model
			region = wavelength in region of interest
			dwin = resolution of model in Angstoms, should be higher than data resolution
			b_win = bounds of plotting window

	Expectations:	None
			 
	'''
	from SRP import create_moog_file_synth
	from SRP import pymoogsilent
	from SRP import moog_synth_chisqr

	for i,e in enumerate(elements):
		t1,t2 = e
		s = np.where(t1 == id_el)
		elements[i] = (t1,x0[s])

	create_moog_file_synth('../DATS/'+selected_model,region-6,region+6,dwin,1.00,b_win,1.1,elements,g_fwhm)
	pymoogsilent()

	#wave_m,flux_m = SRP.read_out3('out3')
	#plt.plot(wave_m,flux_m,'--')

	return moog_synth_chisqr('out3','star.xy',dw)

def pysynth_rot_fxd(x0,id_el,elements,dw,rot,selected_model,region,dwin,b_win):
	'''
	Function:	Wrtie a parameter file for synth module called batch.par, amended to use rotational broadening parameter (fixed)
	Variables:	x0 = abundance values
			id_el = element numbers which are allowed to vary
			elements = full array of tuples, [(element number,abundance)]
			dw = wavelength offset between data and model
			rot = convolve model with rotationally brodened profile
			selected_model = MOOG capable stelalr model
			region = wavelength in region of interest
			dwin = resolution of model in Angstoms, should be higher than data resolution
			b_win = bounds of plotting window

	Expectation:    selected models are in a local directory './DATS/' (probably should make that a var)
			 
	'''

	from SRP import create_moog_file_synth
	from SRP import pymoogsilent
	from SRP import moog_synth_chisqr
    #sanity check
	if (np.max(np.abs(x0)) > 5):
		return 1E3
	#print x0," : x0",rot
	for i,e in enumerate(elements):
		t1,t2 = e
		s = np.where(t1 == id_el)
		#print list(s),id_el,t1
		elements[i] = (t1,x0[s[0][0]])
	
	create_moog_file_synth_rot('../DATS/'+selected_model,region-6,region+6,dwin,1.00,b_win,1.1,elements,rot)
	fyiwdwytm = pymoogsilent()

	#wave_m,flux_m = SRP.read_out3('out3')
	#plt.plot(wave_m,flux_m,'--')

	if fyiwdwytm:
		#jack up chisqr if moog failed
		chisqr = 1E3
	else:
		chisqr = moog_synth_chisqr('out3','star.xy',dw)
	#print chisqr," : chisqr"
	return chisqr
	
def pysynth_rot(x0,id_el,elements,dw,selected_model,region,dwin,b_win):
	'''
	Function:	Wrtie a parameter file for synth module called batch.par, amended to use rotational broadening parameter which can varry in fit
	Variables:	x0 = abundance values
			id_el = element numbers which are allowed to vary
			elements = full array of tuples, [(element number,abundance)]
			dw = wavelength offset between data and model
			rot = convolve model with rotationally brodened profile
			selected_model = MOOG capable stelalr model
			region = wavelength in region of interest
			dwin = resolution of model in Angstoms, should be higher than data resolution
			b_win = bounds of plotting window
			 
	'''

	from SRP import create_moog_file_synth
	from SRP import pymoogsilent
	from SRP import moog_synth_chisqr
	#print x0," : x0 rot"
	rot = x0[0]
	x0 = x0[1:]

	if (rot > 300) | (rot < 0):
		return 10
	if (x0 > 2) | (x0 < -2):
		return 10


	for i,e in enumerate(elements):
		t1,t2 = e
		s = np.where(t1 == id_el)
		elements[i] = (t1,x0[s[0][0]])
	#print elements," : elements"
	create_moog_file_synth_rot('../DATS/'+selected_model,region-6,region+6,dwin,1.00,b_win,1.1,elements,rot)
	fyiwdwytm = pymoogsilent()

	#wave_m,flux_m = SRP.read_out3('out3')
	#plt.plot(wave_m,flux_m,'--')

	if fyiwdwytm:
		#jack up chisqr if moog failed
		chisqr = 1E6
	else:
		chisqr = moog_synth_chisqr('out3','star.xy',dw)
	#print chisqr," : chisqr"
	return chisqr
	
    
def pysynth_rot_metal(x0,id_el,elements,dw,selected_model,region,dwin,b_win):
	'''
	Function:	Wrtie a parameter file for synth module called batch.par, amended to use rotational broadening parameter which can varry in fit, metals a fixed to singular value from solar
	Variables:	x0 = abundance values
			id_el = element numbers which are allowed to vary
			elements = full array of tuples, [(element number,abundance)]
			dw = wavelength offset between data and model
			rot = convolve model with rotationally brodened profile
			selected_model = MOOG capable stelalr model
			region = wavelength in region of interest
			dwin = resolution of model in Angstoms, should be higher than data resolution
			b_win = bounds of plotting window
			 
	'''

	from SRP import create_moog_file_synth
	from SRP import pymoogsilent
	from SRP import moog_synth_chisqr
	print x0," : x0"
	rot = x0[0]/100
	x0 = x0[1]

	if (rot > 300.0/100.0) | (rot < 0.0):
		return 10
	if (x0 > 2) | (x0 < -2):
		return 10

	for i,e in enumerate(elements):
		t1,t2 = e
		elements[i] = (t1,x0)
	#print elements," : elements"
	create_moog_file_synth_rot('../DATS/'+selected_model,region-6,region+6,dwin,1.00,b_win,1.1,elements,rot*100)
	fyiwdwytm = pymoogsilent()

	#wave_m,flux_m = SRP.read_out3('out3')
	#plt.plot(wave_m,flux_m,'--')

	if fyiwdwytm:
		#jack up chisqr if moog failed
		chisqr = 1E6
	else:
		chisqr = moog_synth_chisqr('out3','star.xy',dw)
	#print chisqr," : chisqr"
	return chisqr
	
def pysynth_rot_fxd_metal(x0,id_el,elements,dw,rot,selected_model,region,dwin,b_win):
	'''
	Function:	Wrtie a parameter file for synth module called batch.par, amended to use rotational broadening parameter (fixed)
	Variables:	x0 = abundance values
			id_el = element numbers which are allowed to vary
			elements = full array of tuples, [(element number,abundance)]
			dw = wavelength offset between data and model
			rot = convolve model with rotationally brodened profile
			selected_model = MOOG capable stelalr model
			region = wavelength in region of interest
			dwin = resolution of model in Angstoms, should be higher than data resolution
			b_win = bounds of plotting window

	Expectation:    selected models are in a local directory './DATS/' (probably should make that a var)
			 
	'''

	from SRP import create_moog_file_synth
	from SRP import pymoogsilent
	from SRP import moog_synth_chisqr
    #sanity check
	if (np.max(np.abs(x0)) > 5):
		return 1E3
	#print x0," : x0 fxd",rot
	for i,e in enumerate(elements):
		t1,t2 = e
		elements[i] = (t1,x0[0])
	
	create_moog_file_synth_rot('../DATS/'+selected_model,region-6,region+6,dwin,1.00,b_win,1.1,elements,rot)
	fyiwdwytm = pymoogsilent()

	#wave_m,flux_m = SRP.read_out3('out3')
	#plt.plot(wave_m,flux_m,'--')

	if fyiwdwytm:
		#jack up chisqr if moog failed
		chisqr = 1E3
	else:
		chisqr = moog_synth_chisqr('out3','star.xy',dw)
	#print chisqr," : chisqr"
	return chisqr
	
def pysynth_rot_fxd_mcmc(x0,id_el,elements,dw,rot,selected_model,region,dwin,b_win):
	'''
	Function:	Wrtie a parameter file for synth module called batch.par, amended to use rotational broadening parameter (fixed)
	Variables:	x0 = abundance values
			id_el = element numbers which are allowed to vary
			elements = full array of tuples, [(element number,abundance)]
			dw = wavelength offset between data and model
			rot = convolve model with rotationally brodened profile
			selected_model = MOOG capable stelalr model
			region = wavelength in region of interest
			dwin = resolution of model in Angstoms, should be higher than data resolution
			b_win = bounds of plotting window

	Expectation:    selected models are in a local directory './DATS/' (probably should make that a var)
			 
	'''
	print x0," : x0"
	from SRP import create_moog_file_synth
	from SRP import pymoogsilent
	from SRP import moog_synth_chisqr
    #sanity check
	if (np.max(np.abs(x0)) > 5):
		return 1E3
	#print x0," : x0",rot
	for i,e in enumerate(elements):
		t1,t2 = e
		s = np.where(t1 == id_el)
		#print list(s),id_el,t1
		elements[i] = (t1,x0[s[0][0]])
	
	create_moog_file_synth_rot('../DATS/'+selected_model,region-6,region+6,dwin,1.00,b_win,1.1,elements,rot)
	fyiwdwytm = pymoogsilent()

	#wave_m,flux_m = SRP.read_out3('out3')
	#plt.plot(wave_m,flux_m,'--')

	if fyiwdwytm:
		#jack up chisqr if moog failed
		prob = -np.inf
	else:    
		chisqr = moog_synth_chisqr('out3','star.xy',dw)
		prob = 1/chisqr 
	return np.log(prob)
	
def pysynth_rot_fxd_metal_mcmc(x0,id_el,elements,dw,rot,selected_model,region,dwin,b_win):
	'''
	Function:	Wrtie a parameter file for synth module called batch.par, amended to use rotational broadening parameter (fixed)
	Variables:	x0 = abundance values
			id_el = element numbers which are allowed to vary
			elements = full array of tuples, [(element number,abundance)]
			dw = wavelength offset between data and model
			rot = convolve model with rotationally brodened profile
			selected_model = MOOG capable stelalr model
			region = wavelength in region of interest
			dwin = resolution of model in Angstoms, should be higher than data resolution
			b_win = bounds of plotting window
	Expectation:    selected models are in a local directory './DATS/' (probably should make that a var)
			 
	'''
	print x0," : x0"
	from SRP import create_moog_file_synth
	from SRP import pymoogsilent
	from SRP import moog_synth_chisqr
    #sanity check
	if (np.max(np.abs(x0)) > 5):
		return -np.inf
	#print x0," : x0 fxd",rot
	for i,e in enumerate(elements):
		t1,t2 = e
		elements[i] = (t1,x0[0])
	
	create_moog_file_synth_rot('../DATS/'+selected_model,region-6,region+6,dwin,1.00,b_win,1.1,elements,rot)
	fyiwdwytm = pymoogsilent()
	#wave_m,flux_m = SRP.read_out3('out3')
	#plt.plot(wave_m,flux_m,'--')
	if fyiwdwytm:
		#jack up chisqr if moog failed
		prob = -np.inf
	else:    
		chisqr = moog_synth_chisqr('out3','star.xy',dw)
		prob = 1/chisqr 
	return np.log(prob)
	  
def pysynth_rot_metal_mcmc(x0,args):
	'''
	Function:	Wrtie a parameter file for synth module called batch.par, amended to use rotational broadening parameter which can varry in fit, metals a fixed to singular value from solar
	Variables:	x0 = abundance values
			id_el = element numbers which are allowed to vary
			elements = full array of tuples, [(element number,abundance)]
			dw = wavelength offset between data and model
			rot = convolve model with rotationally brodened profile
			selected_model = MOOG capable stelalr model
			region = wavelength in region of interest
			dwin = resolution of model in Angstoms, should be higher than data resolution
			b_win = bounds of plotting window
			 
	'''
	#print args," : args"
	id_el,elements,dw,selected_model,region,dwin,b_win = args
	#print x0," : x0"
	from SRP import create_moog_file_synth
	from SRP import pymoogsilent
	from SRP import moog_synth_chisqr
	#print x0," : x0"
	rot = x0[0]
	x = x0[1]

	if (rot > 300.0/100.0) | (rot < 0.0):
		return -np.inf
	if (x > 2.0) | (x < -2.0):
		return -np.inf

	for i,e in enumerate(elements):
		t1,t2 = e
		elements[i] = (t1,x)
	#print elements," : elements"
	create_moog_file_synth_rot('../DATS/'+selected_model,region-6,region+6,dwin,1.00,b_win,1.1,elements,rot*100.0)
	fyiwdwytm = pymoogsilent()

	#wave_m,flux_m = SRP.read_out3('out3')
	#plt.plot(wave_m,flux_m,'--')

	if fyiwdwytm:
		#jack up chisqr if moog failed
		prob = -np.inf
	else:    
		chisqr = moog_synth_chisqr('out3','star.xy',dw)
		prob = 1/chisqr 
	return np.log(prob)

def pysynth_rot_metal_mcmc2(x0,args):
	'''
	Function:	Wrtie a parameter file for synth module called batch.par, amended to use rotational broadening parameter which can varry in fit, metals a fixed to singular value from solar
	Variables:	x0 = abundance values
			id_el = element numbers which are allowed to vary
			elements = full array of tuples, [(element number,abundance)]
			dw = wavelength offset between data and model
			rot = convolve model with rotationally brodened profile
			selected_model = MOOG capable stelalr model
			region = wavelength in region of interest
			dwin = resolution of model in Angstoms, should be higher than data resolution
			b_win = bounds of plotting window
			 
	'''
	#print args," : args"
	id_el,elements,dw,selected_model,region,dwin,b_win = args
	#print x0," : x0"
	from SRP import create_moog_file_synth
	from SRP import pymoogsilent
	from SRP import moog_synth_chisqr
	#print x0," : x0"
	rot = x0[0]
	x = x0[1]

	if (rot > 300.0/100.0) | (rot < 0.0):
		return -np.inf,-np.inf
	if (x > 2.0) | (x < -2.0):
		return -np.inf,-np.inf

	for i,e in enumerate(elements):
		t1,t2 = e
		elements[i] = (t1,x)
	#print elements," : elements"
	create_moog_file_synth_rot('../DATS/'+selected_model,region-6,region+6,dwin,1.00,b_win,1.1,elements,rot*100.0)
	fyiwdwytm = pymoogsilent()

	#wave_m,flux_m = SRP.read_out3('out3')
	#plt.plot(wave_m,flux_m,'--')

	if fyiwdwytm:
		#jack up chisqr if moog failed
		proby,probx = -np.inf,-np.inf
		sys.exit()        
	else:    
		chisqr_y,chisqr_x = moog_synth_chisqr_xy('out3','star.xy',dw)
		proby, probx = 1/chisqr_y, 1/chisqr_x 
	return np.log(proby), np.log(probx)

def pysynth_rot_metal_mcmc_lnprob(theta,grad,*args):
	import pickle
	proby0,probx0,laststep = pickle.load(open('prob0.p','r'))
	proby1,probx1 = pysynth_rot_metal_mcmc2(theta,args[0])
	pickle.dump((proby1,probx1,theta),open('prob0.p','w'))
	if np.isfinite(proby1) & np.isfinite(proby0): 
		if ( (proby1-proby0) != 0 | (proby1 > 4) ):
			gradnew = (proby1-proby0)/(theta-np.array(laststep)*0.99)*grad
		else:
			return -10,np.array(grad)            
		#gradnew = (theta-np.array(laststep)*0.99)*grad
	else:
		return -10,np.array(grad)
	#print "theta :",theta,"prob :",(proby1,probx1),"gradnew :",gradnew,"grad :",grad
	#print theta,ivar," : theta,ivar" 
	return proby1,gradnew
    
