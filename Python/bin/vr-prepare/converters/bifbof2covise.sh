
#!/bin/bash

# if called without bif file print usage
if [ $# = "0" ] ; then
  echo "Conversion script was called without a bif file as argument."
  echo
  echo "Usage          : bifbof2covise.sh [-b <boffile>] [-s <scalefactor>] [-o <filename>] [-v <variable>] biffile [biffile ...]"
  echo "Usage(examples): bifbof2covise.sh -b dir1/data.bof *.bif"
  echo "Usage(examples): bifbof2covise.sh -b data.bof -s 0.1 dir2/*.bif"
  echo "Usage(examples): bifbof2covise.sh -b data.bof -o myCocase /home/usr/dir2/1.bif"
  echo "Usage(examples): bifbof2covise.sh -b data.bof -v Temperature 1.bif 2.bif 3.bif"
  echo
  echo "Effect: Convert the bif/bof data to covise-format and put the data into target directory ./CoviseDaten/"
  echo "        Option -b: sets the bof file"
  echo "        Option -s: use a scale factor"
  echo "        Option -o: sets the cocasefile name (if ommited, the name of the bof file is used)"
  echo "        Option -v: sets the name of the variable (if ommited, the name of the bof file is used)"
  exit
fi

# read arguments
cocasename=""
bofname=""
variable=""
scalefactor=1.0
# get options
while getopts "s:o:b:v:" options
do
  case $options in
    b ) bofname=$OPTARG;;
    s ) scalefactor=$OPTARG;;
    o ) cocasename=$OPTARG;;
    v ) variable=$OPTARG;;
  esac
done
shift $(($OPTIND - 1))



# basedir
basedir=`pwd`




# add full path to bofname
if [ "$bofname" != "" ]
   then
      cd $basedir
      bofDir=`dirname $bofname`
      cd $bofDir
      bofDir=`pwd`
      bofname=$bofDir/`basename $bofname`
      cd $basedir
fi


# prepare directory
if [ -d CoviseDaten ]
   then
      echo "Directory CoviseDaten already available"
   else
      mkdir CoviseDaten
      echo "Created directory CoviseDaten"
   fi
cd CoviseDaten
coviseDatenDir=`pwd`
echo "Covisedaten directory is now $coviseDatenDir"




# remove old conversion scripts
if [ -f convert.py ]
then
   echo "convert.py already exists in this directory, removing it"
   rm -f convert.py
fi

# remove old options.inp
if [ -f options.inp ]
then
   echo "options.inp already exists in this directory, removing it"
   rm -f options.inp
fi

# remove old options.inp
if [ -f filenames.inp ]
then
   echo "filenames.inp already exists in this directory, removing it"
   rm -f filenames.inp
fi



# write options into a file

printf "coviseDatenDir = \"" >> options.inp
printf "`echo $coviseDatenDir`" >> options.inp
printf "\"\n" >> options.inp

printf "bofname = \"" >> options.inp
printf "`echo $bofname`" >> options.inp
printf "\"\n" >> options.inp

printf "scaleFactor = " >> options.inp
printf "`echo $scalefactor`" >> options.inp
printf "\n" >> options.inp

printf "cocasename = \"" >> options.inp
printf "`echo $cocasename`" >> options.inp
printf "\"\n" >> options.inp

printf "variable = \"" >> options.inp
printf "`echo $variable`" >> options.inp
printf "\"\n" >> options.inp

cat options.inp >> convert.py



# add bif files

printf "bifFiles = []\n" >> filenames.inp
# loop over arguments
for i in $@; do
   cd $basedir
   # loop over matching files
   for j in `find $i -maxdepth 0 -printf "%p\n"`; do
      #add full path
      cd $basedir
      bifDir=`dirname $j`
      cd $bifDir
      bifDir=`pwd`
      bifName=`basename $j`
      cd $coviseDatenDir
      # write into file
      printf "bifFiles.append(\"" >> filenames.inp
      printf "`echo $bifDir`" >> filenames.inp
      printf "/" >> filenames.inp
      printf "`echo $bifName`" >> filenames.inp
      printf "\")\n" >> filenames.inp
   done
done
cat filenames.inp >> convert.py



# add main code
cat $COVISEDIR/src/application/ui/vr-prepare/converters/BifBof2Covise.py >> convert.py


# GO
covise --script convert.py

