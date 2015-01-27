
#!/bin/bash

#export PATH=$PATH:$COVISEDIR/src/application/ui/vr-prepare
#export PYTHONPATH=$PYTHONPATH:$COVISEDIR/src/application/ui/vr-prepare

# if called without ensight casefile print usage
if [ $# = "0" ] ; then
  echo "Conversion script was called without a casefile as argument"
  echo
  echo "Usage          : ensight2covise.sh [-i <startId>][-s <scalefactor>] [-b] [-p] [-o <filename>] casefile  [casefile ...]"
  echo "Usage          : ensight2covise.sh [-s <scalefactor>] [-b] [-p] casefile"
  echo "Usage(examples): ensight2covise.sh -o mydata mydata.case"
  echo "Usage(examples): ensight2covise.sh -o mydata mydata1.case mydata2.case"
  echo "Usage(examples): ensight2covise.sh mydata.case"
  echo "Usage(examples): ensight2covise.sh EnsightDaten/mydata.case"
  echo "Usage(examples): ensight2covise.sh -s 0.001 mydata.case"
  echo "Usage(examples): ensight2covise.sh -b mydata.case"
  echo "Usage(examples): ensight2covise.sh -s 0.001 -b mydata.case"
  echo "Usage(examples): ensight2covise.sh -p mydata.case"
  echo
  echo "Effect: convert the data referenced by mydata.case to covise-format and"
  echo "        put the data into target directory CoviseDaten."
  echo "        Option -s: use a scalefactor"
  echo "        Option -b: data are byteswapped"
  echo "        Option -p: dataset is originally permas, eqivalent to -s 0.001 -b"
  echo "        Option -o: sets the cocasefile name, if it is ommited it gets the name"
  echo "                   of the ensight case, with this option you can also combine" 
  echo "                   several ensight cases into one cocase" 
  echo "        When directory EnsightDaten is used in the call then put"
  echo "        the target directory CoviseDaten next to the source-directory."
  echo "        Else create directory CoviseDaten as subdirectory."
  exit
fi

scale=1.0
byteswap="FALSE"
cocasename="None"
startId=0
# get options
while getopts "i:s:bpo:" options
do
  case $options in
    i ) startId=$OPTARG;;
    s ) scale=$OPTARG;;
    b ) byteswap="TRUE";;
    p ) byteswap="TRUE"
        scale=0.001;;
    o ) cocasename=$OPTARG;;
  esac
done
shift $(($OPTIND - 1))

#echo "cocasename=$cocasename"
#echo "scale: $scale"
#echo "byteswap: $byteswap"

# get number of parameters
numEnsightCases=$#
#echo numEnsightCases=$numEnsightCases

basedir=`pwd`

# allow more than one parameter only if option -o is set
if [ $numEnsightCases -gt 1 ] 
then 
   if [ "$cocasename" = "None" ]
   then
      echo "if you want to combine several ensight cases in one cocase, please use the option -o to set the cocase name"
      echo "Usage(examples): ensight2covise.sh -o mydata mydata1.case mydata2.case"
      exit
   fi
fi




# create a directory for the covise data
# if we have more than one ensight case file we create a CoviseDaten subdirectory at .
# if we have only one ensight case file
#   if it is in a EnsightDaten subdirectory we create a parallel Covisedaten directory
#   else we create it in the same diretory as in the ensight directory

if [ $numEnsightCases -eq 1 ]
then
   # check if the file is in a ensight directory

   coviseDatenDir=`dirname $1`

   if [ "coviseDatenDir" = "EnsightDaten" ]
   then
      echo "EnsightDaten directory available - make a parallel CoviseDaten directory"
      cd coviseDatenDir
      cd ..
      coviseDatenDir=`pwd`
   else
      echo "EnsightDaten directory not available - make a CoviseDaten subdirectory"
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
   echo "Covisedaten directory is now $coviseDatenDir"




cd $basedir
if [ "$cocasename" = "None" ] 
then
   echo "cocasename not given, take the ensight case name...."
#  fullensightcasename= /data/Kunden/Daimler/SAE/R171/R171.CASE
   fullEnsightCaseName=$1
#  casename=R171.case
   casename=`basename $1`   

   #echo fullEnsightCaseName=$fullEnsightCaseName
   #echo casename=$casename
   # try extensions .case .CASE .encase .encas
   #echo "try extension .case"
   testname=`basename $fullEnsightCaseName .case`
   #echo testname=$testname
   if [ "$testname" != "$casename" ]
   then
      #echo "$testname and $casename are different ! happy"
      casename=$testname
      cocasename=$testname
      #echo "found casename=$casename"
   #  echo ""
   else
      #echo "suffix is not .case"
      #echo "try extension .CASE"
      testname=`basename $fullEnsightCaseName .CASE`
      #echo testname=$testname

      if [ "$testname" != "$casename" ]
      then
         casename=$testname
         cocasename=$testname
         #echo "found casename=$casename"
   #     echo ""
      else
   #     echo "suffix is not .CASE"
   #     echo "try extension .encase"
         testname=`basename $fullEnsightCaseName .encase`
   #     echo testname=$testname
         if [ "$testname" != "$casename" ]
         then
            casename=$testname
            cocasename=$testname
   #        echo "found casename=$casename"
         else
   #        echo "suffix is not .encase"
   #        echo "try extension .encas"
            testname=`basename $fullEnsightCaseName .encas`
   #        echo testname=$testname
            if [ "$testname" != "$casename" ]
               then
                  casename=$testname
                  cocasename=$testname
   #              echo "found casename=$casename"
   #            else
   #              echo "suffix is not .encas"
            fi
         fi
      fi
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

printf "byteswap = " >> options.inp
printf "\"" >> options.inp
printf "`echo $byteswap`" >> options.inp
printf "\"\n" >> options.inp

printf "startId = " >> options.inp
printf "\"" >> options.inp
printf "`echo $startId`" >> options.inp
printf "\"\n" >> options.inp

# append the options to the conversions script
cat options.inp >> convert.py




# add definitions and initialisation stuff
cat $COVISEDIR/Python/bin/vr-prepare/converters/Ensight2CoviseBegin.py >> convert.py

echo ""
echo "loop..."
echo ""

# for all casefiles
for i in $@;
do

   # create filenames.inp

   cd $basedir 
   fullEnsightCaseName=$i
   ensightDatenDir=`dirname $fullEnsightCaseName`
   cd $ensightDatenDir
   #echo pwd=$PWD   
   ensightDatenDir=`pwd`
   cd -
   ensightCaseName=`basename $fullEnsightCaseName`

   #remove old filenames.inp
   cd $coviseDatenDir
   #echo pwd=$PWD  
   #echo ls=`ls`
   if [ -f filenames.inp ]
   then
      echo "filenames.inp already exists in this directory, removing it"
      rm -f filenames.inp
   fi

   printf "fullEnsightCaseName = " >> filenames.inp
   printf "\"" >> filenames.inp
   printf "`echo $ensightDatenDir`" >> filenames.inp
   printf "/" >> filenames.inp
   printf "`echo $ensightCaseName`" >> filenames.inp
   printf "\"\n" >> filenames.inp

   cat filenames.inp $COVISEDIR/Python/bin/vr-prepare/converters/Ensight2CoviseProcessCase.py  >> convert.py

done



cat $COVISEDIR/Python/bin/vr-prepare/converters/Ensight2CoviseEnd.py >> convert.py


covise --script convert.py
