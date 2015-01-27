#!/bin/bash

# small script to pack a COVISE demo
# net-file + all needed data files
# arguments:
# $1 ... COVISE map (*.net) - path & filename
# $2 ... output directory

# check number of arguments
if [ "$#" -ne 2 ]; then
        echo "2 arguments needed (1: COVISE map, 2: output directory)"
        exit
fi
# check existence of output directory
if [ -d $2 ]; then
	outdir=$2
else
	echo "output directory does not exist!"
	exit
fi

# the pure map filename
netfile=`echo $1 | sed 's|/.*/||g'`

# copy all files in map to outdir
echo "copying all files in map $netfile to $outdir"
for i in `grep '^/' $1 | grep -v foo.euc`;do 
	# special treatment for Ensight-files (case-file contains further files ...)
	if [ `echo $i | grep "case" ` ]; then
		ens_copyfiles_short=`echo $i | sed 's|/.*/||g;s|case|*|g'`
		ensight_name=`echo $i | sed 's|case|*|g'`
		# only copy if file does not already exist and is not identical
		cmp $ensight_name $outdir/$ens_copyfiles_short &> /dev/null
		if [ ! $? -eq 0 ]; then         # Test exit status of "cmp" command.
			# not identical: copy!
			echo "   $i";
			echo "      Ensight-File! copying all Ensight files ($ens_copyfiles_short)"
			cp $ensight_name $outdir
		fi
	else
		filename=`echo $i | sed 's|/.*/||g'`
		# only copy if file does not already exist and is not identical
		cmp $i $outdir/$filename &> /dev/null
		if [ ! $? -eq 0 ]; then         # Test exit status of "cmp" command.
			# not identical: copy!
			echo "   $i";
			cp $i $outdir;
		fi
	fi
	done

# copy map to outdir
echo "copying map file $netfile to $outdir"
cp $1 $outdir

# replace file paths in map
echo "replacing file paths in map file"
sed -i "s|/.*/|${outdir}/|g" $outdir/$netfile
