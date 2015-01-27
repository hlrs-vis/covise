#!/bin/bash

#export PATH=$PATH:$COVISEDIR/src/application/ui/vr-prepare
#export PYTHONPATH=$PYTHONPATH:$COVISEDIR/src/application/ui/vr-prepare

# if called without ensight casefile print usage
if [ $# = "0" ] ; then
  echo "Conversion script was called without a casefile as argument"
  echo
  echo "Usage          : ensight2covise.sh [-s <scalefactor>] [-b] casefile [-p] "
  echo "Usage(examples): ensight2covise.sh sourcedata.case"
  echo "Usage(examples): ensight2covise.sh EnsightDaten/sourcedata.case"
  echo "Usage(examples): ensight2covise.sh -s 0.001 sourcedata.case"
  echo "Usage(examples): ensight2covise.sh -b sourcedata.case"
  echo "Usage(examples): ensight2covise.sh -s 0.001 -b sourcedata.case"
  echo "Usage(examples): ensight2covise.sh -p sourcedata.case"
  echo
  echo "Effect: convert the data referenced by mydata.case to covise-format and"
  echo "        put the data into target directory CoviseDaten."
  echo "        When directory EnsightDaten is used in the call then put"
  echo "        the target directory CoviseDaten next to the source-directory."
  echo "        Else create directory CoviseDaten as subdirectory."
  echo "        Option -p (dataset is originally permas) is eqivalent to -s 0.001 -b"
  exit
fi

scale=1.0
byteswap="FALSE"

while getopts ":s:bp" options
do
  case $options in
    s ) scale=$OPTARG;;
    b ) byteswap="TRUE";;
    p ) byteswap="TRUE"
        scale=0.001;;
  esac
done
shift $(($OPTIND - 1))

fullEnsightCaseName=$1
echo "scale: $scale"
echo "byteswap: $byteswap"

echo fullEnsightCaseName=$fullEnsightCaseName
ensightDatenDir=`dirname $fullEnsightCaseName`
echo ensightDatenDir=$ensightDatenDir
ensightCaseName=`basename $fullEnsightCaseName`
echo ensightCaseName=$ensightCaseName
#set -x
# default
casename=$ensightCaseName

# try extensions .case .CASE .encase .encas
#echo "try extension .case"
testname=`basename $fullEnsightCaseName .case`
#echo testname=$testname
if [ "$testname" != "$casename" ]
then
#  echo "$testname and $casename are different ! happy"
   casename=$testname
#  echo "found casename=$casename"
#  echo ""
else
#  echo "suffix is not .case"
#  echo "try extension .CASE"
   testname=`basename $fullEnsightCaseName .CASE`
#  echo testname=$testname

   if [ "$testname" != "$casename" ]
   then
      casename=$testname
#     echo "found casename=$casename"
#     echo ""
   else
#     echo "suffix is not .CASE"
#     echo "try extension .encase"
      testname=`basename $fullEnsightCaseName .encase`
#     echo testname=$testname
      if [ "$testname" != "$casename" ]
      then
         casename=$testname
#        echo "found casename=$casename"
      else
#        echo "suffix is not .encase"
#        echo "try extension .encas"
         testname=`basename $fullEnsightCaseName .encas`
#        echo testname=$testname
         if [ "$testname" != "$casename" ]
            then
               casename=$testname
#              echo "found casename=$casename"
#            else
#              echo "suffix is not .encas"
         fi
      fi
   fi
fi
echo "*********casename=$casename"


# check if the file is in a ensight directory
if [ "$ensightDatenDir" = "EnsightDaten" ]
then
   echo "EnsightDaten directory available - make a parallel CoviseDaten directory"
   cd $ensightDatenDir
   ensightDatenDir=`pwd`
   cd ..
else
   echo "EnsightDaten directory not available - make a CoviseDaten subdirectory"
   cd $ensightDatenDir
   ensightDatenDir=`pwd`
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
echo "Covisedaten directory is now $coviseDatenDir"

# remove old filenames.inp
if [ -f filenames.inp ]
then
   echo "filenames.inp already exists in this directory, removing it"
   rm -f filenames.inp
fi

# create the first part of the python conversion script
printf "fullEnsightCaseName = " >> filenames.inp
printf "\"" >> filenames.inp
printf "`echo $ensightDatenDir`" >> filenames.inp
printf "/" >> filenames.inp
printf "`echo $ensightCaseName`" >> filenames.inp
printf "\"\n" >> filenames.inp

printf "coviseDatenDir = " >> filenames.inp
printf "\"" >> filenames.inp
printf "`echo $coviseDatenDir`" >> filenames.inp
printf "\"\n" >> filenames.inp

printf "casename = " >> filenames.inp
printf "\"" >> filenames.inp
printf "`echo $casename`" >> filenames.inp
printf "\"\n" >> filenames.inp

printf "scale = " >> filenames.inp
printf "`echo $scale`" >> filenames.inp
printf "\n" >> filenames.inp

printf "byteswap = " >> filenames.inp
printf "\"" >> filenames.inp
printf "`echo $byteswap`" >> filenames.inp
printf "\"\n" >> filenames.inp

# cat the template script Ensight2CoviseGeo.py
if [ -f convert.py ]
then
   echo "convert.py already exists in this directory, removing it"
   rm -f convert.py
fi

cat filenames.inp $COVISEDIR/src/application/ui/vr-prepare/Ensight2CoviseGeo.py >> convert.py
covise --script convert.py
