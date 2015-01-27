
#!/bin/bash

#export PATH=$PATH:$COVISEDIR/src/application/ui/vr-prepare
#export PYTHONPATH=$PYTHONPATH:$COVISEDIR/src/application/ui/vr-prepare

# if called without tecplot file print usage
if [ $# = "0" ] ; then
  echo "Conversion script was called without tecplot file as argument"
  echo
  echo "Usage          : tecplot2covise.sh [-s <scalefactor in z>]  [-w <water surface offset>] [-o <filename>] casefile  [casefile ...]"
  echo "Usage          : tecplot2covise.sh [-s <scalefactor>] [-w <water surface offset>] casefile"
  echo "Usage(examples): tecplot2covise.sh -o mydata.plt"
  echo "Usage(examples): tecplot2covise.sh -o mydata mydata.plt"
  echo "Usage(examples): tecplot2covise.sh -o mydata mydata1.plt mydata2.plt"
  echo "Usage(examples): tecplot2covise.sh TecplotDaten/mydata.res"
  echo "Usage(examples): tecplot2covise.sh -s 0.001 mydata.res"
  echo
  echo "Effect: convert the data referenced by mydata.plt to covise-format and"
  echo "        put the data into target directory CoviseDaten."
  echo "        Option -s: use a scalefactor in z direction"
  echo "        Option -o: sets the cocasefile name, if it is ommited it gets the name"
  echo "                   of the plt file, with this option you can also combine" 
  echo "                   several plt files into one cocase" 
  echo "        When directory TecplotDaten is used in the call then put"
  echo "        the target directory CoviseDaten next to the source-directory."
  echo "        Else create directory CoviseDaten as subdirectory."
  exit
fi

scaleZ=1.0
waterSurfaceOffset=1
cocasename="None"

# get options
while getopts "s:w:o:" options
do
  case $options in
    s ) scaleZ=$OPTARG;;
    w ) waterSurfaceOffset=$OPTARG;;
    o ) cocasename=$OPTARG;;
  esac
done
shift $(($OPTIND - 1))

# get name of the script
myName=`basename $0`
echo myName=$myName

if [ "$myName" = "tecplotascii2covise.sh" ]
then
   #echo format2
   format=2
elif [ "$myName" = "tecplotselafin2covise.sh" ]
then
   #echo format3
   format=3
elif [ "$myName" = "tecplotselafinswapped2covise.sh" ]
then
   #echo format7
   format=7

elif [ "$myName" = "tecplot9bin2covise.sh" ]
then
   #echo format4
   format=4
elif [ "$myName" = "tecplot9binswapped2covise.sh" ]
then
   #echo format5
   format=6
elif [ "$myName" = "tecplot10bin2covise.sh" ]
then
   #echo format6
   format=6
fi

#echo "cocasename=$cocasename"
#echo "scaleZ: $scaleZ"

# get number of parameters
numTecplotCases=$#
#echo numTecplotCases=$numTecplotCases

basedir=`pwd`

# allow more than one case only if option -o is set
if [ $numTecplotCases -gt 1 ] 
then 
   if [ "$cocasename" = "None" ]
   then
      echo "if you want to combine several tecplot files in one cocase, please use the option -o to set the cocase name"
      echo "Usage(examples): tecplot2covise.sh -o mydata mydata1.plt mydata2.plt"
      exit
   fi
fi




# create a directory for the covise data
# if we have more than one tecplot case file we create a CoviseDaten subdirectory at .
# if we have only one tecplot case file
#   if it is in a TecplotDaten subdirectory we create a parallel Covisedaten directory
#   else we create it in the same diretory as in the tecplot directory

if [ $numTecplotCases -eq 1 ]
then
   # check if the file is in a tecplot directory

   coviseDatenDir=`dirname $1`

   if [ "coviseDatenDir" = "TecplotDaten" ]
   then
      echo "TecplotDaten directory available - make a parallel CoviseDaten directory"
      cd coviseDatenDir
      cd ..
      coviseDatenDir=`pwd`
   else
      echo "TecplotDaten directory not available - make a CoviseDaten subdirectory"
      cd $coviseDatenDir
      coviseDatenDir=`pwd`
   fi   
else
  coviseDatenDir=`dirname $cocasename`
  cd $coviseDatenDir
fi

if [ -d CoviseDaten ]
   then
      echo "Directory CoviseDaten already available"
   else
      mkdir CoviseDaten
      echo "Created directory CoviseDaten"
   fi
   cd CoviseDaten
   coviseDatenDir=`pwd`
   echo "CoviseDaten directory is now $coviseDatenDir"




cd $basedir
if [ "$cocasename" = "None" ] 
then
   # echo "cocasename not given, take the tecplot name...."
#  fullTecplotCaseName= /data/Kunden/..../name.plt
   fullTecplotCaseName=$1
#  casename=name.res
   casename=`basename $1`   

   #echo fullTecplotCaseName=$fullTecplotCaseName
   #echo casename=$casename
   # try extensions .plt
   #echo "try extension .rplt"
   testname=`basename $fullTecplotCaseName .plt`
   #echo testname=$testname
   if [ "$testname" != "$casename" ]
   then
      #echo "$testname and $casename are different ! happy"
      casename=$testname
      cocasename=$testname
      #echo "found casename=$casename"
   #  echo ""
   else
      echo "suffix is not .plt"
      casename=$testname
      cocasename=$testname       
   fi
else
   casename=$cocasename
fi

cd $coviseDatenDir
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


# write options into a file
printf "coviseDatenDir = " >> options.inp
printf "\"" >> options.inp
printf "`echo $coviseDatenDir`" >> options.inp
printf "\"\n" >> options.inp

printf "cocasename = " >> options.inp
printf "\"" >> options.inp
printf "`echo $cocasename`" >> options.inp
printf "\"\n" >> options.inp

printf "scaleZ = " >> options.inp
printf "`echo $scaleZ`" >> options.inp
printf "\n" >> options.inp

printf "waterSurfaceOffset = " >> options.inp
printf "`echo $waterSurfaceOffset`" >> options.inp
printf "\n" >> options.inp

printf "format = " >> options.inp
printf "\"" >> options.inp
printf "`echo $format`" >> options.inp
printf "\"\n" >> options.inp


# append the options to the conversions script
cat options.inp >> convert.py


# add definitions and initialisation stuff
cat $COVISEDIR/src/application/ui/vr-prepare/converters/Tecplot2CoviseBegin.py >> convert.py

#echo ""
#echo "loop..."
#echo ""

# for all casefiles
for i in $@;
do

   # create filenames.inp

   cd $basedir 
   fullTecplotCaseName=$i
   tecplotDatenDir=`dirname $fullTecplotCaseName`
   cd $tecplotDatenDir
   #echo pwd=$PWD   
   tecplotDatenDir=`pwd`
   cd -
   tecplotCaseName=`basename $fullTecplotCaseName`

   #remove old filenames.inp
   cd $coviseDatenDir
   #echo pwd=$PWD  
   #echo ls=`ls`
   if [ -f filenames.inp ]
   then
      #echo "filenames.inp already exists in this directory, removing it"
      rm -f filenames.inp
   fi

   printf "fullTecplotCaseName = " >> filenames.inp
   printf "\"" >> filenames.inp
   printf "`echo $tecplotDatenDir`" >> filenames.inp
   printf "/" >> filenames.inp
   printf "`echo $tecplotCaseName`" >> filenames.inp
   printf "\"\n" >> filenames.inp

   cat filenames.inp $COVISEDIR/src/application/ui/vr-prepare/converters/Tecplot2CoviseProcessCase.py  >> convert.py

done



cat $COVISEDIR/src/application/ui/vr-prepare/converters/Tecplot2CoviseEnd.py >> convert.py

covise --script convert.py
