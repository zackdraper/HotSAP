#!/bin/csh -f
rm -f fort.*

date

ln -s  /astro/zhd/McD_SpecData/vcsbalm.dat fort.25

#balmer is called and fed with his input control cards
/isluga3/kim/atlas9/bin/balmer9.exe <<EOF
READ PUNCH auto.dat
BEGIN
END
EOF
