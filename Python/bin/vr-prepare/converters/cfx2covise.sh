
#!/bin/bash

#export PATH=$PATH:$COVISEDIR/src/application/ui/vr-prepare
#export PYTHONPATH=$PYTHONPATH:$COVISEDIR/src/application/ui/vr-prepare

# if called without cfx file print usage
if [ $# = "0" ] ; then
  echo "Conversion script was called without cfx file as argument"
  echo
  echo "Usage          : cfx2covise.sh [-i] casefile"
  echo "Usage          : cfx2covise.sh [-v <numVariables>] [-r <resultname> ] [-d <domainname>] [-s <scalefactor>] [-n] [-a] [-p] [-b] [-t] [-c <cocaseFile>] [-o <filename>] casefile  [casefile ...]"
  echo "Usage          : cfx2covise.sh [-v <numVariables>] [-r <resultname> ] [-d <domainname>] [-s <scalefactor>] [-n] [-a] [-p] [-b] [-t] [-c <cocaseFile>] casefile"
  echo "Usage(examples): cfx2covise.sh -v 8 -o mydata mydata.res"
  echo "Usage(examples): cfx2covise.sh -o mydata mydata1.case mydata2.res"
  echo "Usage(examples): cfx2covise.sh mydata.case"
  echo "Usage(examples): cfx2covise.sh CfxDaten/mydata.res"
  echo "Usage(examples): cfx2covise.sh -s 0.001 mydata.res"
  echo "Usage(examples): cfx2covise.sh -i mydata.res"
  echo "Usage(examples): cfx2covise.sh -r myresult -n -c myCoviseCase.cocase mydata.res"
  echo "Usage(examples): cfx2covise.sh -a mydata.res"
  echo "Usage(examples): cfx2covise.sh -b mydata.res"
  echo "Usage(examples): cfx2covise.sh -t mydata.res"
  echo
  echo "Effect: convert the data referenced by mydata.case to covise-format and"
  echo "        put the data into target directory CoviseDaten."
  echo "        Option -i: prints info about the res-file"
  echo "        Option -v: convert only first <numVariables> variables"
  echo "        Option -d: convert only domain <domainname>"
  echo "        Option -r: convert only result <resultname> variables"
  echo "        Option -s: use a scalefactor"
  echo "        Option -o: sets the cocasefile name, if it is ommited it gets the name"
  echo "                   of the cfx file, with this option you can also combine" 
  echo "                   several cfx files into one cocase" 
  echo "        Option -n: do not read the grid"
  echo "        Option -a: additionally create a composed grid of all domains"
  echo "        Option -p: calculate PDYN=PTOT-PRES if PTOT and PRES are present"
  echo "        Option -c: add to cocase file"
  echo "        Option -b: do not read any boundaries"
  echo "        Option -t: forces to read the grid transient"
  echo "        When directory CfxDaten is used in the call then put"
  echo "        the target directory CoviseDaten next to the source-directory."
  echo "        Else create directory CoviseDaten as subdirectory."
  exit
fi

scale=1.0
cocasename="None"
numVariables=8
fixdomain="None"
fixresult="None"
info=0
noGrid=0
composedGrid=0
calculatePDYN=0
cocasefile="None"
readBoundaries=1
transient=0

# get options
while getopts "v:s:o:d:r:inabtc:" options
do
  case $options in
    v ) numVariables=$OPTARG;;
    s ) scale=$OPTARG;;
    o ) cocasename=$OPTARG;;
    d ) fixdomain=$OPTARG;;
    r ) fixresult=$OPTARG;;
    i ) info=1;;
    n ) noGrid=1;;
    a ) composedGrid=1;;
    p ) calculatePDYN=1;;
    c ) cocasefile=$OPTARG;;
    b ) readBoundaries=0;;
    t ) transient=1;;
  esac
done
shift $(($OPTIND - 1))

#echo "cocasename=$cocasename"
#echo "scale: $scale"
#echo "byteswap: $byteswap"

# get number of parameters
numCfxCases=$#
#echo numCfxCases=$numCfxCases

basedir=`pwd`

# allow more than one parameter only if option -o is set
if [ $numCfxCases -gt 1 ] 
then 
   if [ "$cocasename" = "None" ]
   then
      echo "if you want to combine several cfx files in one cocase, please use the option -o to set the cocase name"
      echo "Usage(examples): cfx2covise.sh -o mydata mydata1.res mydata2.res"
      exit
   fi
fi

#allow no grid only if cocasefile is set
if [ $noGrid -eq 1 ]
then
    echo "no grid equal 1"
    if [ "$cocasefile" = "None" ]
    then
        echo "if you want to read only variables without grid, you have to define a cocasefile. Please use optin -c to setthe cocasefile-name"
        echo "Usage(example): cfx2covise.sh -c myCasefile.cocase -n mydata.res"
        exit
    fi
fi


# create a directory for the covise data
# if we have more than one cfx case file we create a CoviseDaten subdirectory at .
# if we have only one cfx case file
#   if it is in a CfxDaten subdirectory we create a parallel Covisedaten directory
#   else we create it in the same diretory as in the cfx directory

if [ $numCfxCases -eq 1 ]
then
   # check if the file is in a cfx directory

   coviseDatenDir=`dirname $1`

   if [ "coviseDatenDir" = "CfxDaten" ]
   then
      echo "CfxDaten directory available - make a parallel CoviseDaten directory"
      cd coviseDatenDir
      cd ..
      coviseDatenDir=`pwd`
   else
      echo "CfxDaten directory not available - make a CoviseDaten subdirectory"
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
   # echo "cocasename not given, take the cfx name...."
#  fullCfxCaseName= /data/Kunden/..../name.res
   fullCfxCaseName=$1
#  casename=name.res
   casename=`basename $1`   

   #echo fullCfxCaseName=$fullCfxCaseName
   #echo casename=$casename
   # try extensions .res
   #echo "try extension .res"
   testname=`basename $fullCfxCaseName .res`
   #echo testname=$testname
   if [ "$testname" != "$casename" ]
   then
      #echo "$testname and $casename are different ! happy"
      casename=$testname
      cocasename=$testname
      #echo "found casename=$casename"
   #  echo ""
   else
      echo "suffix is not .res"
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

printf "scale = " >> options.inp
printf "`echo $scale`" >> options.inp
printf "\n" >> options.inp

printf "numVariables = " >> options.inp
printf "\"" >> options.inp
printf "`echo $numVariables`" >> options.inp
printf "\"\n" >> options.inp

printf "fixdomain = " >> options.inp
printf "\"" >> options.inp
printf "`echo $fixdomain`" >> options.inp
printf "\"\n" >> options.inp

printf "fixresult = " >> options.inp
printf "\"" >> options.inp
printf "`echo $fixresult`" >> options.inp
printf "\"\n" >> options.inp

printf "noGrid = " >> options.inp
printf "\"" >> options.inp
printf "`echo $noGrid`" >> options.inp
printf "\"\n" >> options.inp

printf "composedGrid = " >> options.inp
printf "\"" >> options.inp
printf "`echo $composedGrid`" >> options.inp
printf "\"\n" >> options.inp

printf "calculatePDYN = " >> options.inp
printf "\"" >> options.inp
printf "`echo $calculatePDYN`" >> options.inp
printf "\"\n" >> options.inp

printf "coCaseFile = " >> options.inp
printf "\"" >> options.inp
printf "`echo $cocasefile`" >> options.inp
printf "\"\n" >> options.inp

printf "readBoundaries = " >> options.inp
printf "\"" >> options.inp
printf "`echo $readBoundaries`" >> options.inp
printf "\"\n" >> options.inp

printf "readTransient = " >> options.inp
printf "\"" >> options.inp
printf "`echo $transient`" >> options.inp
printf "\"\n" >> options.inp

# append the options to the conversions script
cat options.inp >> convert.py


# add definitions and initialisation stuff
cat $COVISEDIR/src/application/ui/vr-prepare/converters/Cfx2CoviseBegin.py >> convert.py

#echo ""
#echo "loop..."
#echo ""

# for all casefiles
for i in $@;
do

   # create filenames.inp

   cd $basedir 
   fullCfxCaseName=$i
   cfxDatenDir=`dirname $fullCfxCaseName`
   cd $cfxDatenDir
   #echo pwd=$PWD   
   cfxDatenDir=`pwd`
   cd -
   cfxCaseName=`basename $fullCfxCaseName`

   #remove old filenames.inp
   cd $coviseDatenDir
   #echo pwd=$PWD  
   #echo ls=`ls`
   if [ -f filenames.inp ]
   then
      #echo "filenames.inp already exists in this directory, removing it"
      rm -f filenames.inp
   fi

   printf "fullCfxCaseName = " >> filenames.inp
   printf "\"" >> filenames.inp
   printf "`echo $cfxDatenDir`" >> filenames.inp
   printf "/" >> filenames.inp
   printf "`echo $cfxCaseName`" >> filenames.inp
   printf "\"\n" >> filenames.inp


    if [ $info -eq 1 ]
    then
        cat filenames.inp $COVISEDIR/src/application/ui/vr-prepare/converters/Cfx2CoviseProcessInfo.py  >> convert.py
    else
        cat filenames.inp $COVISEDIR/src/application/ui/vr-prepare/converters/Cfx2CoviseProcessCase.py  >> convert.py
    fi

done


if [ $info -eq 1 ]
then
    cat $COVISEDIR/src/application/ui/vr-prepare/converters/Cfx2CoviseInfoEnd.py >> convert.py
else
    cat $COVISEDIR/src/application/ui/vr-prepare/converters/Cfx2CoviseEnd.py >> convert.py
fi

covise --script convert.py
