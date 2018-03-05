# HotSAP

Overview:

The code has 3 main modules or steps.  One) reduce data from an Echelle Spectrograph to a 1D spectrum. Two) analyze the 1D spectrum for fundemental steallar parameters.  Three) use the 1D spectrum and stellar properties to measure the abundances of certain species.  Each step has been tuned ore developed specifically to handle Hot-type stars (6700 K - 10,000 K) but in a robust way to allow for multiple instruments, data sets, etc.

Dependencies: 

This code requires several leagacy astronomer codes.

MOOG: for stellar abundances.
http://www.as.utexas.edu/~chris/moog.html

BALMER9: for measuring hydrogen blamer lines.
http://kurucz.harvard.edu/programs/balmer/

IRAF: used mainly for CCD data reduction module only.
http://iraf.noao.edu/

IDL: for uvbybeta.pro from NASA IDL library to compute Stromgren parameters (#TODO could be replaced and rewritten with python code)

Numerous other python modules are required as well, but require significantly less trouble to install.

