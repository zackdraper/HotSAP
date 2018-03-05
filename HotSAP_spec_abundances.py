#! /usr/bin/env python

import SRP
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
from scipy.optimize import minimize
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
import glob
import shutil
import csv

from pyhmc import hmc

import multiprocessing as mp

def write_html_abund(star,teff,logg,selected_model,vsini,abund_output,direc):

	f = open(direc+star+'.html','w')
	start = """<html>
	<head>
		<script type="text/javascript">
		var people, asc1 = 1,
			asc2 = 1,
			asc3 = 1;
		window.onload = function () {
			people = document.getElementById("people");
		}
		function sort_table(tbody, col, asc) {
			var rows = tbody.rows,
				rlen = rows.length,
				arr = new Array(),
				i, j, cells, clen;
			// fill the array with values from the table
			for (i = 0; i < rlen; i++) {
				cells = rows[i].cells;
				clen = cells.length;
				arr[i] = new Array();
				for (j = 0; j < clen; j++) {
					arr[i][j] = cells[j].innerHTML;
				}
			}
			// sort the array by the specified column number (col) and order (asc)
			arr.sort(function (a, b) {
				return (a[col] == b[col]) ? 0 : ((a[col] > b[col]) ? asc : -1 * asc);
			});
			// replace existing rows with new rows created from the sorted array
			for (i = 0; i < rlen; i++) {
				rows[i].innerHTML = "<td>" + arr[i].join("</td><td>") + "</td>";
			}
		}
		</script>
		<style type="text/css">
			table {
				border-collapse: collapse;
				border: none;
			}
			th,
			td {
				border: 1px solid black;
				padding: 1px 2px;
				font-family: Times New Roman;
				font-size: 20px;
				text-align: left;
			}
			th {
				background-color: #C8C8C8;
				cursor: pointer;
			}
	</style>    
	</head>
	<body>"""

	end = """</body>
	</html>"""

	f.write(start)

	f.write("<b>Star Name:</b> "+star+'<br />')
	f.write("<b>Effective Temperature: </b>"+teff+'<br/>')
	f.write("<b>Surface Gravity: </b>"+logg+'<br/>')
	f.write("<b>Selected Model: </b>"+selected_model+'<br/>')
	f.write("<b>v sin(i): </b>"+vsini+'<br/>')
	f.write("<b></b>"+'<br/>')

	s_table = "<table>\n<thead>\n"

	table =''
	table += '<th onclick="sort_table(abund, 0, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">Element</th>'
	table += '<th onclick="sort_table(abund, 1, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">Wavelength (A)</th>'
	table += '<th onclick="sort_table(abund, 2, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">eV</th>'
	table += '<th onclick="sort_table(abund, 3, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">loggf</th>'
	table += '<th onclick="sort_table(abund, 4, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">Abundance</th>'
	table += '<th onclick="sort_table(abund, 5, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">Abund + error</th>'
	table += '<th onclick="sort_table(abund, 6, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">Abund - error</th>'
	table += '<th onclick="sort_table(abund, 7, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">delta W</th>'
	table += '<th onclick="sort_table(abund, 8, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">chisqr</th>'
	table += '<th onclick="sort_table(abund, 9, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">Fit Type</th>'
	table += '<th onclick="sort_table(abund, 10, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">EW</th>'
	table += '<th onclick="sort_table(abund, 12, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">minimum</th>'
	table += '<th onclick="sort_table(abund, 13, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">vsin(i)</th>'
	table += '<th onclick="sort_table(abund, 11, asc1); asc1 *= -1; asc2 = 1; asc3 = 1;">Image</th>'
    
	table += '</thead>\n<tbody id=abund>\n'
	for a in abund_output:
		image_link = '<a href=\"'+a[11].replace('..','../..')+'\">Fitted Plot</a>'    
		#print np.hstack((a[0:10],image_link))
		table += "<tr><td>"+"</td><td>".join(np.hstack((np.hstack((a[0:11],a[12:14])),image_link)))+"</td></tr>\n"

	e_table = "</tbody>\n</table>"

	f.write(s_table)
	f.write(table)
	f.write(e_table)
	f.write(end)
	f.close()

def singlet_lines_calc(singlet_linelist):

	worker_id = mp.current_process()
	worker_name = worker_id.name
	#worker_id = list(worker_name)
    
	#wid = int(''.join([s for s in worker_id if s.isdigit()]))
	#n_wid = str(wid)
	#if wid > num_proc:
		#n_wid = str(num_proc - (wid % num_proc))
		#worker_name = worker_name.replace(str(wid),n_wid)

	work_dir = cwd+'/'+str(worker_name)

	print singlet_linelist,worker_name

	vsini = -1
	abund_output = np.array(['','','','','','','','','','','','',str(vsini),''])

	#print np.size(abund_output)

	if not os.path.exists(work_dir):
		print work_dir
		os.makedirs(work_dir)

	os.chdir(work_dir)

	mtype = singlet_linelist[4]
	#print l

#find region

	region = np.float(singlet_linelist[1])

#check if line was fit in previous interation of doublet fitting, it probably was given doublet...

	#if singlet_linelist[1] in tuple(last_regional_linelist[0:,1]):
	#	return

	ids = np.where((rvcorr_lamb < region+10) & (rvcorr_lamb > region-10))
	reg_wav = rvcorr_lamb[ids]
	reg_flx = flux[ids]

#when gaps in data exist at line center, skip
	ids2 = np.where((reg_wav < region+2) & (reg_wav > region-2))
	try:
		ntst = np.size(ids2)/np.size(ids)
	except ZeroDivisionError:
		ntst = 0
	if (ntst > 0.2):
		print "Insuffecient data in selected region"
		return abund_output.flatten()

#start take deivatives of line region
	first_derivative = (np.roll(reg_flx,-1)-np.roll(reg_flx,1)) / (np.roll(reg_wav,-1)-np.roll(reg_wav,1))
	#smooth out noise, for first round of noisey data
	first_derivative = filt.gaussian_filter((first_derivative),sigma=5)
	second_derivative = (np.roll(first_derivative,-1)-np.roll(first_derivative,1)) / (np.roll(reg_wav,-1)-np.roll(reg_wav,1))

#code break to see a specific line in derivatives
	if 'bang' in str(w):
	#if tag == 0:
		plt.clf()
		plt.plot(linewave,first_derivative+1.05,'g')
		plt.plot(linewave,second_derivative+1.1,'c')
		plt.plot(linewave,lineflux,'b')
		plt.plot(linewave,np.abs(lineflux-1)*np.abs(first_derivative)*np.abs(second_derivative),'r')

		condition = (np.abs(first_derivative) < 0.01) & (second_derivative < 0)

		plt.plot(linewave[condition],lineflux[condition],'ro')

		plt.show()
		#sys.exit()

#continuum condition based on derivatives
	condition = (np.abs(first_derivative) < 0.01) & (second_derivative < 0)

	high_points = reg_flx[condition]
	hpts_wave = reg_wav[condition]


	if (np.size(hpts_wave) < 5):
		print "Insuffecient points in derivatives"
		return abund_output.flatten()

	fit = np.polyfit(hpts_wave,high_points,2)

#iterate linear fit
	norm = reg_wav**2*fit[0]+reg_wav*fit[1]+fit[2]
	norm2 = hpts_wave**2*fit[0]+hpts_wave*fit[1]+fit[2]

#renormalize data
	reg_flx = reg_flx/norm
	high_points = high_points/norm2

	#sys.exit()

#find gaussian dispersion from singlelet lines
	g_fwhm = 0.6

#establish line list and observed spectrum for MOOG
	ids = np.where((wlist < region+10) & (wlist > region-10)) 
	regional_linelist = linelist[ids]
	#conform linelist from abund to synth
	regional_linelist[0:,4] = '0.0'

	SRP.write_linelist_moog("regional_linelist.moog",regional_linelist)

	SRP.write_starxy(star,reg_wav,reg_flx)

	dwin = (reg_wav[1]-reg_wav[0])/5

#redo region 
#establish line list and observed spectrum for MOOG
	ids = np.where((wlist < region+2) & (wlist > region-2)) 
	regional_linelist = linelist[ids]
	#conform linelist from abund to synth
	regional_linelist[0:,4] = '0.0'

	SRP.write_linelist_moog("regional_linelist.moog",regional_linelist)

	#ids = np.where((reg_wav < region+2) & (reg_wav > region-2))
	#reg_wav,reg_flx = reg_wav[ids],reg_flx[ids]

	#SRP.write_starxy(star,reg_wav,reg_flx)

	uni_e = []
	for e in regional_linelist[0:,0]:
		uni_e.append(np.floor(np.float(SRP.elementnumber(e))))

	e_list = list(uni_e)
	uni_e = list(set(uni_e))

	elements = []
	for e in uni_e:
		elements.append( ( np.floor(np.float(e)), np.float(0.01) ) )

#this step is because moog is stupid and wont allow a synthesis of a region 10 A blueward of the linelist
#this tricks moog to doing what I tell it to do

	if np.abs(np.float(regional_linelist[0,1])-(region-10)) > 10:
		region_cut = np.float(regional_linelist[0,1])-9
	else:
		region_cut = region-10
    
	SRP.create_moog_file_synth('../DATS/'+selected_model,region_cut,region+10,dwin,1.00,np.min([np.min(reg_flx),0.8]),1.1,elements,g_fwhm)

	SRP.pymoogsilent()

	dw = SRP.xcorr_moog_synth('out3','star.xy',5)
	#print dw

	b_win = np.min([np.min(reg_flx),0.8])

	el_num = np.int(np.float(SRP.elementnumber(singlet_linelist[0])))

#calculate result
	options = {"maxfev": 250, "xtol": (np.size(uni_e)*0.001)}

	min_a = minimize(SRP.pysynth_rot,np.append([60],np.zeros(np.size(uni_e))),args=(uni_e,elements,dw,selected_model,region,dwin,b_win),method='powell',options=options)
	result = min_a['x']

	vsini = min_a["x"][0]
	#print vsini,": vsini"
	if (vsini > 125):
		condition = [np.abs(hpts_wave-region) > 3]
		high_points = high_points[condition]
		hpts_wave = hpts_wave[condition]
		if (np.size(hpts_wave) < 2):
			print "Insuffecient points in derivatives"
			return abund_output.flatten()
		fit = np.polyfit(hpts_wave,high_points,2)
		#iterate linear fit
		norm = reg_wav**2*fit[0]+reg_wav*fit[1]+fit[2]
		norm2 = hpts_wave**2*fit[0]+hpts_wave*fit[1]+fit[2]
		#renormalize data
		reg_flx = reg_flx/norm
		high_points = high_points/norm2
		#print elements," : elements"
		min_a = minimize(SRP.pysynth_rot,np.append([150],np.zeros(np.size(uni_e))),args=(uni_e,elements,dw,selected_model,region,dwin,b_win),method='powell',options=options)
		result = min_a['x']
		vsini = min_a["x"][0]
     
	options = {"maxfev": 250, "xtol": (np.size(uni_e)*0.001)}
	element_results = zip(uni_e,list(result[1:]))
	#print element_results
#find errors
	elements_e1 = elements
	min_a_e1 = minimize(SRP.pysynth_rot_fxd,result[1:],args=(uni_e,elements_e1,dw,vsini,selected_model_error1,region,dwin,b_win),method='powell',options=options)
    
	element_results_e1 = zip(uni_e,list([min_a_e1['x']]))

	elements_e2 = elements
	min_a_e2 = minimize(SRP.pysynth_rot_fxd,result[1:],args=(uni_e,elements_e2,dw,vsini,selected_model_error2,region,dwin,b_win),method='powell',options=options)

	element_results_e2 = zip(uni_e,list([min_a_e2['x']]))

	element_results = zip(element_results,element_results_e1,element_results_e2)

#save a plot of fit
	f = plt.figure()
	ax = f.add_subplot(111)
	wave_m,flux_m = SRP.read_out3('out3')
	plt.plot(wave_m,flux_m,'g-')

	#SRP.pysynth(min_a[],8,elements,dw,g_fwhm,selected_model,region,dwin,b_win)

	plt.plot(reg_wav-dw,reg_flx,'bo')
	plt.plot(hpts_wave,high_points,'ro')
	plt.xlabel('wavelength')
	plt.ylabel('normalized flux')

	plt.ylim([b_win,1.1])
	plt.xlim([np.min(wave_m),np.max(wave_m)])

	plt.title(star+'  '+selected_model+'  '+str(dw))

	plot_name = '../'+star_dir+'synth_'+str(region)+'_'+singlet_linelist[0]+'_'+mtype+'.pdf'

	#crude EW
	ew = str(np.sum(np.abs(flux_m-np.median(flux_m))*(wave_m[1]-wave_m[0])))
	line_abund = -99
	line_error1 = -99
	line_error2 = -99
	min_flux = np.min(flux_m)
    
	#print np.size(np.append(regional_linelist[0,0:4],[("%.3f" % 9),("%.3f" % 9),("%.3f" % 9),dw,min_a['fun'],mtype,plot_name] ) )
	for r in regional_linelist:
		for ea,ea1,ea2 in element_results:
			if ea[0] == np.int(np.float(SRP.elementnumber(r[0]))):
				line_abund = ea[1]
				line_error1 = ea1[1]
				line_error2 = ea2[1]

		plt.axvline(np.float(r[1]),0.0,1.5)
		plt.text(np.float(r[1])-0.25, 1.09, r[0]+" = "+("%.3f" % line_abund), rotation=90)
		#print r[0:4],line_abund,line_error1,line_error2,dw,min_a['fun'],mtype,ew,plot_name,min_flux,vsini

		abund_output = np.vstack( ( abund_output,np.append(r[0:4],[("%.3f" % line_abund),("%.3f" % line_error1),("%.3f" % line_error2),dw,min_a['fun'],mtype,ew,plot_name,min_flux,vsini] ) ) )

	plt.savefig(plot_name)
	plt.close(f)

	last_regional_linelist = regional_linelist

	abund_output = abund_output[1:,0:]

	return abund_output.flatten()

def multiplet_lines_calc(multiplet_line,fixed_vsini,mg4481_special = False):

	abund_output = np.array([['','','','','','','','','','','','','','']])

	worker_id = mp.current_process()
	worker_name = worker_id.name
	#worker_id = list(worker_name)
    
	#wid = int(''.join([s for s in worker_id if s.isdigit()]))
	#n_wid = str(wid)
	#if wid > num_proc:
		#n_wid = str(num_proc - (wid % num_proc))
		#worker_name = worker_name.replace(str(wid),n_wid)

	work_dir = cwd+'/'+str(worker_name)
    
	print multiplet_line,worker_name

	if not os.path.exists(work_dir):
		print work_dir
		os.makedirs(work_dir)

	os.chdir(work_dir)

	mtype = multiplet_line[4]

#find region

	region = np.float(multiplet_line[1])

#check if line was fit in previous interation of doublet fitting, it probably was given doublet...

	#if l[1] in tuple(last_regional_linelist[0:,1]):
	#	continue

	ids = np.where((rvcorr_lamb < region+10) & (rvcorr_lamb > region-10))
	reg_wav = rvcorr_lamb[ids]
	reg_flx = flux[ids]

#when gaps in data exist at line center, skip
	ids2 = np.where((reg_wav < region+2) & (reg_wav > region-2))
	try:
		ntst = np.size(ids2)/np.size(ids)
	except ZeroDivisionError:
		ntst = 0
	if (ntst > 0.2):
		print "Insuffecient data in selected region"
		return abund_output.flatten()

	#plt.close('all')
	#f = plt.figure()
	#ax = f.add_subplot(111)
	#plt.plot(reg_wav,reg_flx)
	#plt.show()

#start take deivatives of line region
	first_derivative = (np.roll(reg_flx,-1)-np.roll(reg_flx,1)) / (np.roll(reg_wav,-1)-np.roll(reg_wav,1))
	#smooth out noise, for first round of noisey data
	first_derivative = filt.gaussian_filter((first_derivative),sigma=5)
	second_derivative = (np.roll(first_derivative,-1)-np.roll(first_derivative,1)) / (np.roll(reg_wav,-1)-np.roll(reg_wav,1))

#code break to see a specific line in derivatives
	if 'bang' in str(w):
	#if tag == 0:
		plt.clf()
		plt.plot(linewave,first_derivative+1.05,'g')
		plt.plot(linewave,second_derivative+1.1,'c')
		plt.plot(linewave,lineflux,'b')
		plt.plot(linewave,np.abs(lineflux-1)*np.abs(first_derivative)*np.abs(second_derivative),'r')

		condition = (np.abs(first_derivative) < 0.01) & (second_derivative < 0)

		plt.plot(linewave[condition],lineflux[condition],'ro')

		plt.show()
		#sys.exit()

#continuum condition based on derivatives
	condition = (np.abs(first_derivative) < 0.01) & (second_derivative < 0)

	high_points = reg_flx[condition]
	hpts_wave = reg_wav[condition]
    
	if (fixed_vsini > 125):
		condition = [np.abs(hpts_wave-region) > 3]
		high_points = high_points[condition]
		hpts_wave = hpts_wave[condition]

	if (np.size(hpts_wave) < 5):
		print "Insuffecient data in derivative selection"
		return abund_output.flatten()

	fit = np.polyfit(hpts_wave,high_points,2)

#iterate linear fit
	norm = reg_wav**2*fit[0]+reg_wav*fit[1]+fit[2]
	norm2 = hpts_wave**2*fit[0]+hpts_wave*fit[1]+fit[2]

#renormalize data
	reg_flx = reg_flx/norm
	high_points = high_points/norm2

	#sys.exit()

#find gaussian dispersion from singlelet lines
	g_fwhm = 0.6


	#try:
	if (0 == 0):
#redo region 
#establish line list and observed spectrum for MOOG
		ids = np.where((wlist < region+4) & (wlist > region-4)) 
		regional_linelist = linelist[ids]
	#conform linelist from abund to synth
		regional_linelist[0:,4] = '0.0'

		SRP.write_linelist_moog("regional_linelist.moog",regional_linelist)

	#ids = np.where((reg_wav < region+2.5) & (reg_wav > region-2.5))
	#reg_wav,reg_flx = reg_wav[ids],reg_flx[ids]
		dwin = (reg_wav[1]-reg_wav[0])/5

		SRP.write_starxy(star,reg_wav,reg_flx)

		uni_e = []
		for e in regional_linelist[0:,0]:
			uni_e.append(np.floor(np.float(SRP.elementnumber(e))))

		e_list = list(uni_e)
		uni_e = list(set(uni_e))

		elements = []
		for e in uni_e:
			elements.append( ( np.floor(np.float(e)), np.float(0.01) ) )


#redo region

#this step is because moog is stupid and wont allow a synthesis of a region 10 A blueward of the linelist
#this tricks moog to doing what I tell it to do

		if np.abs(np.float(regional_linelist[0,1])-(region-10)) > 10:
			region_cut = np.float(regional_linelist[0,1])-9
		else:
			region_cut = region-10

		SRP.create_moog_file_synth_rot('../DATS/'+selected_model,region_cut,region+10,dwin,1.00,np.min([np.min(reg_flx),0.8]),1.1,elements,fixed_vsini)

		SRP.pymoogsilent()

		dw = SRP.xcorr_moog_synth('out3','star.xy',5)
	#print dw

		b_win = np.min([np.min(reg_flx),0.8])

		el_num = np.int(np.float(SRP.elementnumber(multiplet_line[0])))

		elements = []
		for e in uni_e:
			elements.append( (np.floor(e),0.01) )
            
		func = SRP.pysynth_rot_fxd
		func2 = SRP.pysynth_rot_fxd    
		input_a = np.zeros(np.size(uni_e),dtype=np.float)
		args = (uni_e,elements,dw,fixed_vsini,selected_model,region,dwin,b_win)
            
		if (fixed_vsini > 125):
			#uni_e 
			func = SRP.pysynth_rot_metal
			func2 = SRP.pysynth_rot_fxd_metal            
			input_a = [125.0,0.01]
			args = (uni_e,elements,dw,selected_model,region,dwin,b_win)
			options = {"maxfev": 250, "xtol":0.001, "direc":(np.diag([10,0.1]))}
			min_a = minimize(func,input_a,args=args,method='powell',options=options)
			input_a = min_a['x']
			options = {"maxfev": 250, "xtol":0.001, "direc":(np.diag([-10,-0.1]))}
			min_a = minimize(func,input_a,args=args,method='powell',options=options)
			input_a = min_a['x']
			print "going special mode metallicity"            
		#print func       
		fixed_vsini_l = np.float(fixed_vsini)
#calculate result

		options = {"maxfev": 250, "xtol":0.001}
		min_a = minimize(func,input_a,args=args,method='powell',options=options)
		result = min_a['x']
		if ((fixed_vsini > 125)):
			fixed_vsini_l = min_a['x'][0]
			result = min_a['x'][1:][0]            
		#print min_a
		#sys.exit()
		if np.size(result) == 1:
			#print uni_e,result," : e res" 
			element_results = zip(uni_e,[result])
		else:
			element_results = zip(uni_e,result)
            
		if (fixed_vsini > 125):
			element_results = zip(uni_e,np.zeros(np.size(uni_e))+result)
            
		#sys.exit()    

#find errors

		min_a_e1 = minimize(func2,result,args=(uni_e,elements,dw,fixed_vsini_l,selected_model_error1,region,dwin,b_win),method='powell',options=options)
		result = min_a_e1['x']
		#if (fixed_vsini > 125):
			#result = min_a_e1['x'][1:]    
            
		if np.size(result) == 1:
			#print uni_e,min_a_e1['x']," : e res" 
			element_results_e1 = zip(uni_e,[result])
		else:
			element_results_e1 = zip(uni_e,result)
            
		if (fixed_vsini > 125):
			element_results_e1 = zip(uni_e,np.zeros(np.size(uni_e))+result)
            
		min_a_e2 = minimize(func2,result,args=(uni_e,elements,dw,fixed_vsini_l,selected_model_error2,region,dwin,b_win),method='powell',options=options)
		result = min_a_e2['x']
		#if (fixed_vsini > 125):
			#result = min_a_e2['x'][1:] 
            
		if np.size(result) == 1:
			#print uni_e,min_a_e2['x']," : e res" 
			element_results_e2 = zip(uni_e,[result])      
		else:
			element_results_e2 = zip(uni_e,result)
            
		if (fixed_vsini > 125):
			element_results_e2 = zip(uni_e,np.zeros(np.size(uni_e))+result)
            
		element_results = zip(element_results,element_results_e1,element_results_e2)
        
		print element_results
		       
#save a plot of fit
		f = plt.figure()
		ax = f.add_subplot(111)
		wave_m,flux_m = SRP.read_out3('out3')
		plt.plot(wave_m,flux_m,'g-')

	#SRP.pysynth(min_a[],8,elements,dw,g_fwhm,selected_model,region,dwin,b_win)

		plt.plot(reg_wav-dw,reg_flx,'bo')
		plt.plot(hpts_wave,high_points,'ro')

		plt.xlabel('wavelength')
		plt.ylabel('normalized flux')

		plt.ylim([b_win,1.1])
		plt.xlim([np.min(wave_m),np.max(wave_m)])

		plt.title(star+'  '+selected_model+'  '+str(dw))

		#crude EW
		ew = str(np.sum(np.abs(flux_m-np.median(flux_m))*(wave_m[1]-wave_m[0])))   
		min_flux = np.min(flux_m)
    
		plot_name = '../'+star_dir+'synth_'+str(region)+'_'+multiplet_line[0]+'_'+mtype+'.pdf'
		for r in regional_linelist:
			line_abund = -99
			line_error1 = -99
			line_error2 = -99

			for ea,ea1,ea2 in element_results:
				if ea[0] == np.int(np.float(SRP.elementnumber(r[0]))):
					line_abund = ea[1]
					line_error1 = ea1[1]
					line_error2 = ea2[1]

			plt.axvline(np.float(r[1]),0.0,1.5)
			plt.text(np.float(r[1])-0.25, 1.09, r[0]+" = "+("%.3f" % line_abund), rotation=90)
			#print abund_output
			#print r[0:4],line_abund,line_error1,line_error2,dw,min_a['fun'],mtype,ew,plot_name,min_flux,fixed_vsini_l
			abund_output = np.vstack( ( abund_output,np.append(r[0:4],[("%.3f" % line_abund),("%.3f" % line_error1),("%.3f" % line_error2),dw,min_a['fun'],mtype,ew,plot_name,min_flux,fixed_vsini_l] ) ) )

		plt.savefig(plot_name)
		plt.close(f)

		#last_regional_linelist = regional_linelist
		abund_output = abund_output[1:,0:]

		return abund_output.flatten()
	#except TypeError:
		#return abund_output.flatten()
        
def multiplet_lines_calc_mcmc(multiplet_line,fixed_vsini,mg4481_special = False):

	abund_output = np.array([['','','','','','','','','','','','','','']])

	worker_id = mp.current_process()
	worker_name = worker_id.name
	#worker_id = list(worker_name)
    
	#wid = int(''.join([s for s in worker_id if s.isdigit()]))
	#n_wid = str(wid)
	#if wid > num_proc:
		#n_wid = str(num_proc - (wid % num_proc))
		#worker_name = worker_name.replace(str(wid),n_wid)

	work_dir = cwd+'/'+str(worker_name)
    
	print multiplet_line,worker_name

	if not os.path.exists(work_dir):
		print work_dir
		os.makedirs(work_dir)

	os.chdir(work_dir)

	mtype = multiplet_line[4]

#find region

	region = np.float(multiplet_line[1])

#check if line was fit in previous interation of doublet fitting, it probably was given doublet...

	#if l[1] in tuple(last_regional_linelist[0:,1]):
	#	continue

	ids = np.where((rvcorr_lamb < region+10) & (rvcorr_lamb > region-10))
	reg_wav = rvcorr_lamb[ids]
	reg_flx = flux[ids]

#when gaps in data exist at line center, skip
	ids2 = np.where((reg_wav < region+2) & (reg_wav > region-2))
	try:
		ntst = np.size(ids2)/np.size(ids)
	except ZeroDivisionError:
		ntst = 0
	if (ntst > 0.2):
		print "Insuffecient data in selected region"
		return abund_output.flatten()

	#plt.close('all')
	#f = plt.figure()
	#ax = f.add_subplot(111)
	#plt.plot(reg_wav,reg_flx)
	#plt.show()

#start take deivatives of line region
	first_derivative = (np.roll(reg_flx,-1)-np.roll(reg_flx,1)) / (np.roll(reg_wav,-1)-np.roll(reg_wav,1))
	#smooth out noise, for first round of noisey data
	first_derivative = filt.gaussian_filter((first_derivative),sigma=5)
	second_derivative = (np.roll(first_derivative,-1)-np.roll(first_derivative,1)) / (np.roll(reg_wav,-1)-np.roll(reg_wav,1))

#code break to see a specific line in derivatives
	if 'bang' in str(w):
	#if tag == 0:
		plt.clf()
		plt.plot(linewave,first_derivative+1.05,'g')
		plt.plot(linewave,second_derivative+1.1,'c')
		plt.plot(linewave,lineflux,'b')
		plt.plot(linewave,np.abs(lineflux-1)*np.abs(first_derivative)*np.abs(second_derivative),'r')

		condition = (np.abs(first_derivative) < 0.01) & (second_derivative < 0)

		plt.plot(linewave[condition],lineflux[condition],'ro')

		plt.show()
		#sys.exit()

#continuum condition based on derivatives
	condition = (np.abs(first_derivative) < 0.01) & (second_derivative < 0)

	high_points = reg_flx[condition]
	hpts_wave = reg_wav[condition]
    
	if (fixed_vsini > 125):
		condition = [np.abs(hpts_wave-region) > 3]
		high_points = high_points[condition]
		hpts_wave = hpts_wave[condition]

	if (np.size(hpts_wave) < 5):
		print "Insuffecient data in derivative selection"
		return abund_output.flatten()

	fit = np.polyfit(hpts_wave,high_points,2)

#iterate linear fit
	norm = reg_wav**2*fit[0]+reg_wav*fit[1]+fit[2]
	norm2 = hpts_wave**2*fit[0]+hpts_wave*fit[1]+fit[2]

#renormalize data
	reg_flx = reg_flx/norm
	high_points = high_points/norm2

	#sys.exit()

#find gaussian dispersion from singlelet lines
	g_fwhm = 0.6

	#try:
	if (0 == 0):
#redo region 
#establish line list and observed spectrum for MOOG
		ids = np.where((wlist < region+4) & (wlist > region-4)) 
		regional_linelist = linelist[ids]
	#conform linelist from abund to synth
		regional_linelist[0:,4] = '0.0'

		SRP.write_linelist_moog("regional_linelist.moog",regional_linelist)

	#ids = np.where((reg_wav < region+2.5) & (reg_wav > region-2.5))
	#reg_wav,reg_flx = reg_wav[ids],reg_flx[ids]
		dwin = (reg_wav[1]-reg_wav[0])/5

		SRP.write_starxy(star,reg_wav,reg_flx)

		uni_e = []
		for e in regional_linelist[0:,0]:
			uni_e.append(np.floor(np.float(SRP.elementnumber(e))))

		e_list = list(uni_e)
		uni_e = list(set(uni_e))

		elements = []
		for e in uni_e:
			elements.append( ( np.floor(np.float(e)), np.float(0.01) ) )


#redo region

#this step is because moog is stupid and wont allow a synthesis of a region 10 A blueward of the linelist
#this tricks moog to doing what I tell it to do

		if np.abs(np.float(regional_linelist[0,1])-(region-10)) > 10:
			region_cut = np.float(regional_linelist[0,1])-9
		else:
			region_cut = region-10

		SRP.create_moog_file_synth_rot('../DATS/'+selected_model,region_cut,region+10,dwin,1.00,np.min([np.min(reg_flx),0.8]),1.1,elements,fixed_vsini)

		SRP.pymoogsilent()

		dw = SRP.xcorr_moog_synth('out3','star.xy',5)
	#print dw

		b_win = np.min([np.min(reg_flx),0.8])

		el_num = np.int(np.float(SRP.elementnumber(multiplet_line[0])))

		elements = []
		for e in uni_e:
			elements.append( (np.floor(e),0.01) )
            
		func = SRP.pysynth_rot_metal_mcmc_lnprob
		func2 = SRP.pysynth_rot_fxd_metal            
		args = [uni_e,elements,dw,selected_model,region,dwin,b_win]
		inputs = [[fixed_vsini/100.0,-1.0],[fixed_vsini/100.0,-0.5],[fixed_vsini/100.0,0.0],[fixed_vsini/100.0,0.5],[fixed_vsini/100.0,1.0]]           
            
		tst1 = func(np.array(inputs[0]),np.array([-0.5,-0.25]),args)
		tst2 = func(np.array(inputs[1]),np.array([-0.5,-0.25]),args)
		tst3 = func(np.array(inputs[2]),np.array([-0.5,-0.25]),args)
		tst4 = func(np.array(inputs[3]),np.array([-0.5,-0.25]),args) 
		tst5 = func(np.array(inputs[4]),np.array([-0.5,-0.25]),args) 
		tst_arry = [tst1[0],tst2[0],tst3[0],tst4[0],tst5[0]]
		print tst_arry
		idx = np.where(np.max(tst_arry) == tst_arry)
		print idx
		if np.size(idx) > 1:
			idx = list(idx)[0]  
		print idx
		input_a = inputs[idx[0]]
		print input_a
            
			#p0 = ((np.random.rand(ndim*nwalkers).reshape((nwalkers, ndim))*0.25) * np.array(input_a)) - np.array([0,-0.125])  
			#sampler = emcee.EnsembleSampler(nwalkers,ndim,func,args=args) 
			            
		print "going special mode metallicity"            
		#print func       
#calculate result
		#pos, prob, state = sampler.run_mcmc(p0,100)
		#print pos, prob, state, " : sampler" 
		pickle.dump((0.01,0.01,[0.01,0.01]),open('prob0.p','w'))
		samples, logp = hmc(func,input_a,args=([0.25,0.1],args),epsilon=0.1,n_steps=5,n_samples=100,return_logp=True,n_burn=10)
		samples = samples[logp > 0]       
		logp = logp[logp > 0]
		weights = 1/logp**2
		print weights
		if np.sum(weights) == 0 | (np.size(weights) == 0):
			weights = np.zeros(np.size(weights))+1
		try:
			result0 = np.average(samples,axis=0,weights=weights)
		except:
			result0 = -99            
		print result0," : average mean"
		maxid = np.where(logp == np.max(logp))     
		#print maxid     
		result0 = np.mean(samples[list(maxid)],axis=0) 
		options = {"maxfev": 250, "xtol": (np.size(uni_e)*0.001), "direc":([0.1])}  
        
		fixed_vsini_l = result0[0]*100
		result0 = result0[1:] 
        
		min_a = minimize(func2,result0,args=(uni_e,elements,dw,fixed_vsini_l,selected_model,region,dwin,b_win),method='powell',options=options)
		result0 = min_a['x']
            
		element_results = zip(uni_e,np.zeros(np.size(uni_e))+result0) 

#find errors

		min_a_e1 = minimize(func2,result0,args=(uni_e,elements,dw,fixed_vsini_l,selected_model_error1,region,dwin,b_win),method='powell',options=options)
		result = np.array(result0)
		result = min_a_e1['x']    
            
		element_results_e1 = zip(uni_e,np.zeros(np.size(uni_e))+result)
            
		min_a_e2 = minimize(func2,result0,args=(uni_e,elements,dw,fixed_vsini_l,selected_model_error2,region,dwin,b_win),method='powell',options=options)
		result = min_a_e2['x'] 
            
		element_results_e2 = zip(uni_e,np.zeros(np.size(uni_e))+result)
            
		element_results = zip(element_results,element_results_e1,element_results_e2)
        
		#print element_results
		       
#save a plot of fit
		f = plt.figure()
		ax = f.add_subplot(111)
		final_chi = func(np.array([fixed_vsini_l/100,result0]),np.array([-0.5,-0.25]),args)
		wave_m,flux_m = SRP.read_out3('out3')
		plt.plot(wave_m,flux_m,'g-')

	#SRP.pysynth(min_a[],8,elements,dw,g_fwhm,selected_model,region,dwin,b_win)

		plt.plot(reg_wav-dw,reg_flx,'bo')
		plt.plot(hpts_wave,high_points,'ro')

		plt.xlabel('wavelength')
		plt.ylabel('normalized flux')

		plt.ylim([b_win,1.1])
		plt.xlim([np.min(wave_m),np.max(wave_m)])

		plt.title(star+'  '+selected_model+'  '+str(dw))

		#crude EW
		ew = str(np.sum(np.abs(flux_m-np.median(flux_m))*(wave_m[1]-wave_m[0])))   
		min_flux = np.min(flux_m)
    
		plot_name = '../'+star_dir+'synth_'+str(region)+'_'+multiplet_line[0]+'_'+mtype+'.pdf'
		for r in regional_linelist:
			line_abund = -99
			line_error1 = -99
			line_error2 = -99

			for ea,ea1,ea2 in element_results:
				if ea[0] == np.int(np.float(SRP.elementnumber(r[0]))):
					line_abund = ea[1]
					line_error1 = ea1[1]
					line_error2 = ea2[1]

			plt.axvline(np.float(r[1]),0.0,1.5)
			plt.text(np.float(r[1])-0.25, 1.09, r[0]+" = "+("%.3f" % line_abund), rotation=90)
			#print abund_output
			#print r[0:4],line_abund,line_error1,line_error2,dw,min_a['fun'],mtype,ew,plot_name,min_flux,fixed_vsini_l
			abund_output = np.vstack( ( abund_output,np.append(r[0:4],[("%.3f" % line_abund),("%.3f" % line_error1),("%.3f" % line_error2),dw,1/(np.exp(final_chi[0])),mtype,ew,plot_name,min_flux,fixed_vsini_l] ) ) )

		plt.savefig(plot_name)
		plt.close(f)

		#last_regional_linelist = regional_linelist
		abund_output = abund_output[1:,0:]

		return abund_output.flatten()
	#except TypeError:
		#return abund_output.flatten()
        
def multiplet_lines_calc_manual(multiplet_line,fixed_vsini,input_a):

	abund_output = np.array([['','','','','','','','','','','','','','']])

	worker_id = mp.current_process()
	worker_name = worker_id.name
	#worker_id = list(worker_name)
    
	#wid = int(''.join([s for s in worker_id if s.isdigit()]))
	#n_wid = str(wid)
	#if wid > num_proc:
		#n_wid = str(num_proc - (wid % num_proc))
		#worker_name = worker_name.replace(str(wid),n_wid)

	work_dir = cwd+'/'+str(worker_name)
    
	print multiplet_line,worker_name

	if not os.path.exists(work_dir):
		print work_dir
		os.makedirs(work_dir)

	os.chdir(work_dir)

	mtype = multiplet_line[4]

#find region

	region = np.float(multiplet_line[1])

#check if line was fit in previous interation of doublet fitting, it probably was given doublet...

	#if l[1] in tuple(last_regional_linelist[0:,1]):
	#	continue

	ids = np.where((rvcorr_lamb < region+10) & (rvcorr_lamb > region-10))
	reg_wav = rvcorr_lamb[ids]
	reg_flx = flux[ids]

#when gaps in data exist at line center, skip
	ids2 = np.where((reg_wav < region+2) & (reg_wav > region-2))
	try:
		ntst = np.size(ids2)/np.size(ids)
	except ZeroDivisionError:
		ntst = 0
	if (ntst > 0.2):
		print "Insuffecient data in selected region"
		return abund_output.flatten()

	#plt.close('all')
	#f = plt.figure()
	#ax = f.add_subplot(111)
	#plt.plot(reg_wav,reg_flx)
	#plt.show()

#start take deivatives of line region
	first_derivative = (np.roll(reg_flx,-1)-np.roll(reg_flx,1)) / (np.roll(reg_wav,-1)-np.roll(reg_wav,1))
	#smooth out noise, for first round of noisey data
	first_derivative = filt.gaussian_filter((first_derivative),sigma=5)
	second_derivative = (np.roll(first_derivative,-1)-np.roll(first_derivative,1)) / (np.roll(reg_wav,-1)-np.roll(reg_wav,1))

#code break to see a specific line in derivatives
	if 'bang' in str(w):
	#if tag == 0:
		plt.clf()
		plt.plot(linewave,first_derivative+1.05,'g')
		plt.plot(linewave,second_derivative+1.1,'c')
		plt.plot(linewave,lineflux,'b')
		plt.plot(linewave,np.abs(lineflux-1)*np.abs(first_derivative)*np.abs(second_derivative),'r')

		condition = (np.abs(first_derivative) < 0.01) & (second_derivative < 0)

		plt.plot(linewave[condition],lineflux[condition],'ro')

		plt.show()
		#sys.exit()

#continuum condition based on derivatives
	condition = (np.abs(first_derivative) < 0.01) & (second_derivative < 0)

	high_points = reg_flx[condition]
	hpts_wave = reg_wav[condition]
    
	if (fixed_vsini > 125):
		condition = [np.abs(hpts_wave-region) > 3]
		high_points = high_points[condition]
		hpts_wave = hpts_wave[condition]

	if (np.size(hpts_wave) < 5):
		print "Insuffecient data in derivative selection"
		return abund_output.flatten()

	fit = np.polyfit(hpts_wave,high_points,2)

#iterate linear fit
	norm = reg_wav**2*fit[0]+reg_wav*fit[1]+fit[2]
	norm2 = hpts_wave**2*fit[0]+hpts_wave*fit[1]+fit[2]

#renormalize data
	reg_flx = reg_flx/norm
	high_points = high_points/norm2

	#sys.exit()

#find gaussian dispersion from singlelet lines
	g_fwhm = 0.6

	#try:
	if (0 == 0):
#redo region 
#establish line list and observed spectrum for MOOG
		ids = np.where((wlist < region+4) & (wlist > region-4)) 
		regional_linelist = linelist[ids]
	#conform linelist from abund to synth
		regional_linelist[0:,4] = '0.0'

		SRP.write_linelist_moog("regional_linelist.moog",regional_linelist)

	#ids = np.where((reg_wav < region+2.5) & (reg_wav > region-2.5))
	#reg_wav,reg_flx = reg_wav[ids],reg_flx[ids]
		dwin = (reg_wav[1]-reg_wav[0])/5

		SRP.write_starxy(star,reg_wav,reg_flx)

		uni_e = []
		for e in regional_linelist[0:,0]:
			uni_e.append(np.floor(np.float(SRP.elementnumber(e))))

		e_list = list(uni_e)
		uni_e = list(set(uni_e))

		elements = []
		for e in uni_e:
			elements.append( ( np.floor(np.float(e)), np.float(0.01) ) )


#redo region

#this step is because moog is stupid and wont allow a synthesis of a region 10 A blueward of the linelist
#this tricks moog to doing what I tell it to do

		if np.abs(np.float(regional_linelist[0,1])-(region-10)) > 10:
			region_cut = np.float(regional_linelist[0,1])-9
		else:
			region_cut = region-10

		SRP.create_moog_file_synth_rot('../DATS/'+selected_model,region_cut,region+10,dwin,1.00,np.min([np.min(reg_flx),0.8]),1.1,elements,fixed_vsini)

		SRP.pymoogsilent()

		dw = SRP.xcorr_moog_synth('out3','star.xy',5)
	#print dw

		b_win = np.min([np.min(reg_flx),0.8])

		el_num = np.int(np.float(SRP.elementnumber(multiplet_line[0])))

		elements = []
		for e in uni_e:
			elements.append( (np.floor(e),0.01) )
            
		func = SRP.pysynth_rot_metal_mcmc_lnprob
		func2 = SRP.pysynth_rot_fxd_metal            
		args = [uni_e,elements,dw,selected_model,region,dwin,b_win]
		inputs = [[fixed_vsini/100.0,-1.0],[fixed_vsini/100.0,-0.5],[fixed_vsini/100.0,0.0],[fixed_vsini/100.0,0.5],[fixed_vsini/100.0,1.0]]           
		print input_a            
		tst1 = func(input_a,np.array([-0.5,-0.25]),args)
        
		fixed_vsini_l = input_a[0]*100
		result0 = input_a[1]
        
		options = {"maxfev": 250, "xtol": (np.size(uni_e)*0.001), "direc":([0.1])}  
            
		element_results = zip(uni_e,np.zeros(np.size(uni_e))+result0) 

#find errors

		min_a_e1 = minimize(func2,result0-0.5,args=(uni_e,elements,dw,fixed_vsini_l,selected_model_error1,region,dwin,b_win),method='powell',options=options)
		result = np.array(result0)
		result = min_a_e1['x']    
            
		element_results_e1 = zip(uni_e,np.zeros(np.size(uni_e))+result)
            
		min_a_e2 = minimize(func2,result0-0.5,args=(uni_e,elements,dw,fixed_vsini_l,selected_model_error2,region,dwin,b_win),method='powell',options=options)
		result = min_a_e2['x'] 
            
		element_results_e2 = zip(uni_e,np.zeros(np.size(uni_e))+result)
            
		element_results = zip(element_results,element_results_e1,element_results_e2)
        
		#print element_results
		       
#save a plot of fit
		f = plt.figure()
		ax = f.add_subplot(111)
		final_chi = func(np.array([fixed_vsini_l/100,result0]),np.array([-0.5,-0.25]),args)
		wave_m,flux_m = SRP.read_out3('out3')
		plt.plot(wave_m,flux_m,'g-')

	#SRP.pysynth(min_a[],8,elements,dw,g_fwhm,selected_model,region,dwin,b_win)

		plt.plot(reg_wav-dw,reg_flx,'bo')
		plt.plot(hpts_wave,high_points,'ro')

		plt.xlabel('wavelength')
		plt.ylabel('normalized flux')

		plt.ylim([b_win,1.1])
		plt.xlim([np.min(wave_m),np.max(wave_m)])

		plt.title(star+'  '+selected_model+'  '+str(dw))

		#crude EW
		ew = str(np.sum(np.abs(flux_m-np.median(flux_m))*(wave_m[1]-wave_m[0])))   
		min_flux = np.min(flux_m)
    
		plot_name = '../'+star_dir+'synth_'+str(region)+'_'+multiplet_line[0]+'_'+mtype+'.pdf'
		for r in regional_linelist:
			line_abund = -99
			line_error1 = -99
			line_error2 = -99

			for ea,ea1,ea2 in element_results:
				if ea[0] == np.int(np.float(SRP.elementnumber(r[0]))):
					line_abund = ea[1]
					line_error1 = ea1[1]
					line_error2 = ea2[1]

			plt.axvline(np.float(r[1]),0.0,1.5)
			plt.text(np.float(r[1])-0.25, 1.09, r[0]+" = "+("%.3f" % line_abund), rotation=90)
			#print abund_output
			#print r[0:4],line_abund,line_error1,line_error2,dw,min_a['fun'],mtype,ew,plot_name,min_flux,fixed_vsini_l
			abund_output = np.vstack( ( abund_output,np.append(r[0:4],[("%.3f" % line_abund),("%.3f" % line_error1),("%.3f" % line_error2),dw,1/(np.exp(final_chi[0])),mtype,ew,plot_name,min_flux,fixed_vsini_l] ) ) )

		plt.savefig(plot_name)
		plt.close(f)

		#last_regional_linelist = regional_linelist
		abund_output = abund_output[1:,0:]

		return abund_output.flatten()
	#except TypeError:
		#return abund_output.flatten()
if __name__ == "__main__":

###########################################################################################################################################################
# Stellar Abundances Pipeline 


	#data_dir = 'McD_raw_data/McD_data/apr24/'
	data_dir = 'raw_data/'

	#output_dir = 'McD_pipe_output_apr24/'
	output_dir = 'pipe_output/'

	cntnorm_dir = output_dir+'Continuum_Normalized/' 

	ew_dir = output_dir+'Abundances_EW/'

	temp_dir = output_dir+'Abundances_EW/'
    
	html_dir = output_dir+'HTML/'

	#stellar_param_file = output_dir+'pipe_table_2016-08-09-11:38:28.pickle'
	#stellar_param_file = output_dir+'pipe_table_2016-08-09-17:09:37.pickle'
	#stellar_param_file = output_dir+'pipe_table_2016-11-15-18:38:05.pickle'
	#stellar_param_file = output_dir+'pipe_table_2018-02-26-15:43:47.pickle'
	stellar_param_file = output_dir+'pipe_table_2018-03-01-14:07:29.pickle'

###########################################################################################################################################################

#Make directories if not present

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	if not os.path.exists(cntnorm_dir):
		os.makedirs(cntnorm_dir)

	if not os.path.exists(ew_dir):
		os.makedirs(ew_dir)

	if not os.path.exists(temp_dir):
		os.makedirs(temp_dir)
        
	if not os.path.exists(html_dir):
		os.makedirs(html_dir)

	files = os.listdir(cntnorm_dir+'.')
	flats=[]
	dark=[]
	lamp=[]
	obj=[]
	other=[]
	names=[]

	timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
	

	#pipe_table=[]

#mark bad files

	#files = list(set(files)-set(["dl60232.fits","dl60233.fits","dl60234.fits"]))

#data sorting

	#sort files by basic types and target names
	for i in files:
		if '.fits' in i and 'cntnorm' in i and 'legendre' in i:
			obj.append(i)
		elif True:
			other.append(i)

#target processing

	targets = [o.replace('_cntnorm_legendre.fits','') for o in obj]

	#extract spectrum from each star

	beastmode=raw_input('Beast Mode ? (y/n): ').lower()
	beast = beastmode in "y"

	#skip to a star in particular
	if beast:
		findstar = "n"
	else:
		findstar=raw_input('Looking for a star in particular ? (star catalog id): ').lower()

	if findstar == "n":
		print "Looping through all stars"
		
	else:
		#sys.exit()
		ids = np.array([findstar in x.lower() for x in targets])        
		if np.sum(ids) == 0:
			print "Star not found"
			print targets
			sys.exit()
		else:
			obj = list(np.array(obj)[ids])
			targets = list(np.array(targets)[ids])
			print targets        

	pipe_start = time.time()

	try:
		f = open(stellar_param_file, "r" )
		star_param = pickle.load(f)
		f.close()

	except IOError:
		print "stellar parameter files doesnt exist"

#get model files for moog

	dats_dir = os.listdir('./DATS/')

	datfiles_mmod = ['model',5000,2.0]
	for x in dats_dir: 
		if 'mmod' in x:
			model_params = str.split(x,'_')
			datfiles_mmod = np.vstack((datfiles_mmod,[x,model_params[1],model_params[3].replace(".mmod","")]))

	datfiles_mmod = datfiles_mmod[1:,0:]

	model_teff = datfiles_mmod[0:,1]
	model_teff = np.array([float(i) for i in model_teff])
	model_logg = datfiles_mmod[0:,2]
	model_logg = np.array([float(i) for i in model_logg])/100

	for f,star in zip(obj,targets):
		start = time.time()
		#print f
		#star = f.replace("_cntnorm_legendre.fits","")
		print star
		star_data1 = 0
		star_hdr1 = 0

#find temperature and log(g)

		#use pipeline file
		#print star_param
		if np.shape(star_param)[0] > 1:
			id_star = list(np.where(star_param[0:,0] == star))
		else:
			id_star = list(np.where(star_param[0][0] == star))
            
		if np.size(id_star) == 0:
		
			if (star == "Vega_Takeda"):
				teff = 9550
				logg = 3.95

			elif (star == "Procyon"):
				teff = 6554
				logg = 4.00
			elif True:
				print "star not found: "+star
				continue
			
		else:
			id_star = id_star[0]
			try:
				star_row = star_param[id_star][0]
				teff = np.float(star_row[9])
				logg = np.float(star_row[11])
			except IndexError:
				star_row = star_param[id_star]
				teff = np.float(star_row[9])
				logg = np.float(star_row[11])
			#balmer profile determined teff and logg
            
		if (logg > 4.5):
			logg = 4.5  #unrealistic 5.0 logg sent back a model grid to 4.5 typical of main sequence          


		#find closest match to models list
		idx = (np.abs(model_teff-teff)+1000*np.abs(model_logg-logg)).argmin()
		selected_model = datfiles_mmod[idx][0]

		print teff,logg,selected_model,f

		idx = (np.abs(model_teff-(teff+250))+1000*np.abs(model_logg-(logg+0.5))).argmin()
		selected_model_error1 = datfiles_mmod[idx][0]

		idx = (np.abs(model_teff-(teff-250))+1000*np.abs(model_logg-(logg-0.5))).argmin()
		selected_model_error2 = datfiles_mmod[idx][0]

		#sys.exit()
#spectral analysis	
		if(beast):
			proc2 = "y"
		else:
			proc2=raw_input('Proceed with abundance analysis of spectrum '+star+'? (y/n): ').lower()
		

		if proc2 in "y":
	
			star_hdu1 = pyfits.open(cntnorm_dir+f)
			star_data1 = star_hdu1[0].data
			star_hdr1 = star_hdu1[0].header
			star_hdu1.close()

			if (np.shape(star_data1)[0] > 2):
				rvcorr_lamb = star_data1[0:,0]
				flux = star_data1[0:,1]
			else:
				rvcorr_lamb = star_data1[0,0:]
				flux = star_data1[1,0:]

			flux = filt.median_filter((flux),size=3)
			flux = filt.gaussian_filter((flux),sigma=1)
			#sys.exit()            
		#find metal line list

			#Takeda Vega Atlas / currated
			f = open("./linelist_special.pickle", "r" )
			linelist = pickle.load(f)
			f.close()

		#sort list by wavelength

			znumlist = np.array([linelist[0:,1].astype(np.float)])
			ids = znumlist.argsort()
			linelist = linelist[ids,0:][0]
			
		#presort by multiplicity

			dw = 4
			wlist = (linelist[0:,1]).astype(np.float)

			#need to add another column
			col = np.full( (np.size(linelist[0:,0]),1),'0.0' )
			linelist = np.c_[linelist,col]

			popt_list = np.array([0,0,0,0])

			for tag,l in enumerate(linelist):
	
				w = np.float(l[1])
				e = l[0]
            
				if True:
					# boundary exception
					#if (np.abs(tag - np.floor(sz/2)) < np.floor(sz/2)):
					bid = np.where( (wlist > (w-dw)) & (wlist < (w+dw)) )

					if (np.size(bid) == 3):
						mtype = "triplet"
					elif (np.size(bid) == 2):
						mtype = "doublet"
					elif (np.size(bid) == 1):
						mtype = "singlet"
					elif (np.size(bid) > 3):
						mtype = "fubar"
                        
				#exceptional case of Mg2
				if (np.abs(w-4481) < 2):
					mtype = "fubar"
					if (w == 4481.126):
						mtype = "doublet"
                        
				if (np.abs(w-6158) < 4):
					mtype = "fubar"
					if (w == 6158.187):
						mtype = "singlet"

				linelist[tag,4] = mtype

			EW_data = np.array([0,0,0,0,0,0,0,0])
			
			eVlist = (linelist[0:,2]).astype(np.float)

		#use singlets to measure line widths

			singlet_linelist = np.where(linelist[0:,4] == "singlet")
			singlet_linelist = linelist[singlet_linelist,0:][0]
			#singlet_linelist = singlet_linelist[np.array([(i % 2) == 1 for i in np.arange(np.shape(singlet_linelist)[0])])]
			singlet_linelist_sav = np.array(singlet_linelist)
            
			doublet_linelist = np.where(linelist[0:,4] == "doublet")
			doublet_linelist = linelist[doublet_linelist,0:][0]
			doublet_linelist = doublet_linelist[np.array([(i % 2) == 1 for i in np.arange(np.shape(doublet_linelist)[0])])]
			triplet_linelist = np.where(linelist[0:,4] == "triplet")
			triplet_linelist = linelist[triplet_linelist,0:][0]
			triplet_linelist = triplet_linelist[np.array([(i % 2) == 1 for i in np.arange(np.shape(triplet_linelist)[0])])]

			vsini_linelist = singlet_linelist[0,0:]
			mg4481_linelist = doublet_linelist[0,0:]
			i=0
			for tag,l in enumerate(singlet_linelist_sav):
				w = np.float(l[1])
				if (w in [4468.507,4501.273,4583.837,5018.440,6371.371,6347.109]):
					#print tag
					vsini_linelist = np.vstack((vsini_linelist,l)) 
					singlet_linelist = np.vstack( (singlet_linelist[0:(tag-i),0:],singlet_linelist[(tag-i+1):,0:]) )
					i+=1
			i=0
			for tag,l in enumerate(doublet_linelist):
				w = l[1]                    
				if (w in ['4481.126']):
					mg4481_linelist = np.vstack((mg4481_linelist,l))
					doublet_linelist = np.vstack( (doublet_linelist[0:(tag-i),0:],doublet_linelist[(tag-i+1):,0:]) )
					i+=1
					#print tag                 
			vsini_linelist = vsini_linelist[1:,0:]
			mg4481_linelist = mg4481_linelist[1:,0:]
			#sys.exit()
		#start singlet analysis
			dw = 13
			#sys.exit()
			star_dir = ew_dir+star.replace('*','')+'/'

			if not os.path.exists(star_dir):
				os.makedirs(star_dir)

#Lets go parrallel!

			cwd = os.getcwd()
			num_proc = 32


			fixed_vsini = 63
			#print sys.argv[0]
			#multiplet_lines_calc_manual(mg4481_linelist[0],198,[np.float(sys.argv[1]),np.float(sys.argv[2])])
			#sys.exit()
            
			final_output = np.array(['','','','','','','','','','','','','',''])
			pool = mp.Pool(num_proc)
            
#Vsini linelist first
			vsini_results = [pool.apply_async(singlet_lines_calc, [m]) for m in vsini_linelist]
     
			for i,x in enumerate(vsini_results):
				print x
				try:
					res = x.get()
					final_output = np.vstack((final_output,res.reshape(np.size(res)/14,14)))
				except (RuntimeError,TypeError,IndexError,ValueError):
					print "I feel like being an asshole and wont retrieve this: ",vsini_linelist[i]                    
                
			final_output = final_output[final_output[0:,0] != '']
			vsini_table = final_output[0:,-1].astype('float')
			fixed_vsini = np.nanmedian(vsini_table)
			#pool.close()
			#sys.exit()
			singlet_results = [pool.apply_async(singlet_lines_calc, [m]) for m in singlet_linelist]
     
			for i,x in enumerate(singlet_results):
				print x
				try:
					res = x.get()
					final_output = np.vstack((final_output,res.reshape(np.size(res)/14,14)))
				except (RuntimeError,TypeError,IndexError,ValueError):
					print "I feel like being an asshole and wont retrieve this: ",singlet_linelist[i]                    
                
			final_output = final_output[final_output[0:,0] != '']

#Lets go parrallel!
			
			#res = multiplet_lines_calc_mcmc(mg4481_linelist[0],15,True)
			#fixed_vsini = res[-1].astype('float')
			#print res
			#os.chdir(cwd)
			#sys.exit()
			#mg4481_special = True
#compensate for large vsini to get Mg
			if (fixed_vsini > 100):
				try:
					res = multiplet_lines_calc_mcmc(mg4481_linelist[0],fixed_vsini,mg4481_special = True)
					final_output = np.vstack((final_output,res.reshape(np.size(res)/14,14)))
					fixed_vsini = res[-1].astype('float')
				except (TypeError,IndexError,ValueError):
					print "I feel like being an asshole and wont retrieve this: ",mg4481_linelist[0]
			else:
				try:
					res = multiplet_lines_calc(mg4481_linelist[0],fixed_vsini)
					fixed_vsini = np.nanmedian(vsini_table)
				except (TypeError,IndexError,ValueError):
					print "I feel like being an asshole and wont retrieve this: ",mg4481_linelist[0]
			os.chdir(cwd)
			pool.close()
			pool.join()                
			#sys.exit()
#Lets go parrallel!
			#mg4481_special = False
			#sys.exit() 
			pool = mp.Pool(num_proc)
			cwd = os.getcwd()
			#fixed_vsini = 150
			#write_html_abund(star,str(teff),str(logg),selected_model,str("%.3f" % fixed_vsini),abund_output,html_dir)
			#sys.exit()
	#start measuring doublets and triplets


			#par_output_2 = pool.map_async(multiplet_lines_calc,np.vstack((doublet_linelist,triplet_linelist)))
			#par_output_result_2 = par_output_2.get()
			#for x in par_output_result_2:
				#final_output = np.vstack((final_output,res.reshape(np.size(res)/12,12)))
			#donequeue = mp.Queue()
			multiplet_linelist = np.vstack((doublet_linelist,triplet_linelist))
			#multiplet_lines_calc(mg4481_linelist[0])
			#sys.exit()
			multiple_results = [pool.apply_async(multiplet_lines_calc, [m,fixed_vsini]) for m in multiplet_linelist]

			for i,x in enumerate(multiple_results):
				#print x
				try:
					res = x.get()  
					new_arr = res.reshape(np.size(res)/14,14)
					final_output = np.vstack((final_output,new_arr))
				except (RuntimeError,IndexError,ValueError):
					print "I feel like being an asshole and wont retrieve this: ",multiplet_linelist[i]                    
				#print res.reshape(np.size(res)/12,12)

			final_output = final_output[final_output[0:,0] != '']
                    
			#abund_output = abund_output[1:,0:]
			pool.close()         
			pool.join() 
            
#Lets go parrallel!

			final_output = {'abundances':final_output, 'star':star, 'teff':teff, 'logg':logg, 'selected_model':selected_model, 'vsini':fixed_vsini}

			pickle.dump(final_output,open(ew_dir+star+"_abund_"+timestamp+".pickle","w"))

            #visual aid for data output
			write_html_abund(star,str(teff),str(logg),selected_model,str("%.3f" % fixed_vsini),final_output["abundances"],html_dir)

			end = time.time()
			print "elapsed time for star: ",np.int(end-start)," sec"
			if beast:
				os.system('rm -r PoolWorker-*')
			#sys.exit()
			

		elif proc in "n":
			print "Skipping star"
		elif proc in "b":
			break
		else:
			print "Bugged input"		
	
		end = time.time()

	print "elapsed time for pipeline: ",np.int(end-pipe_start)," sec"

#clean up
	#os.remove('master_dark.fits')

