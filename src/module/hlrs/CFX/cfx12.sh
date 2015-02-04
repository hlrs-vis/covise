#!/bin/bash

export DISPLAY=:0.0
export CFX5_UNITS_DIR=/mnt/raid/soft/cfx/v121/CFX/etc
export CFX_HOME=/mnt/raid/soft/cfx/v121
export PATH=$CFX_HOME/CFX/bin:$PATH
export LD_LIBRARY_PATH=$CFX_HOME/CFX/tools/mpich-1.2.6-1/Linux-amd64/lib/shared:$LD_LIBRARY_PATH
#export LM_LICENSE_FILE=28000@hwwsys2.hww.de:28000@galba.hlrs.de
#export LM_LICENSE_FILE=26000@galba.hlrs.de:28000@galba.hlrs.de:50030@bond.hlrs.de
for f in `ipcs -s | awk '{ print $2 }'`; do ipcrm -s $f; done
#$HOME/bin/clean_semaphores.sh
#ssh viscluster02 $HOME/bin/clean_semaphores.sh

killall solver-pvm.exe
killall solver-mpich.exe

echo Anzahl der Argumente: $#
i=0;
while (($#));do
 case $1 in
        numProc)
           numProc=$2
           shift
           shift
          ;;
	Hostlist)
	   hostlist=$2
         shift
         shift
	   ;;
	CO_SIMLIB_CONN)
	   export CO_SIMLIB_CONN=$2
	   shift
           shift
	   ;;
       MachineType)
          machineType=$2
          shift
          shift
          ;;
	startup)
	   startup=$2
	   shift
	   shift
             ;;
	revolutions)
	   revolutions=$2
	   shift
	   shift
	   ;;
	deffile)
	  deffile=$2
	  shift
	  shift
	  ;;
	maxIterations)
	  maxIterations=$2
          shift
	  shift
	  ;;
	locationString)
	  locationString=$2
	  shift
	  shift
     ;;
   *)
     echo "unsupported argument $2"
     shift
     shift
     ;;
esac

#export CO_SIMLIB_CONN=$1


done

echo numProc=$numProc
echo Hostlist=$hostlist
echo MachineType=$machineType
echo startup=$startup
echo revolutions=$revolutions
echo deffile=$deffile
echo maxIterations=$maxIterations
echo locationString=$locationString

locationString=`echo ${locationString} | sed -e "s/\"//g" `
echo locationString=$locationString


echo "cfx12.sh: CO_SIMLIB_CONN: $CO_SIMLIB_CONN"

#CFXDIR=$HOME/covise/src/application/hlrs/CFX
CFXDIR=/data/CFX

killall /mnt/raid/soft/cfx/v121/CFX/bin/cfx5solve
rm -rf ~/CombustorEDM_0*


#hier CFX starten und dieses Beispiel mit dem Solver loesen
#/mnt/raid/home/hpcbenni/cfx_sub/user_import

#cfx5solve -def ~/cfx_sub/CombustorEDM/CombustorEDM.def  -par-local -partition 2
#rm /mnt/raid/home/hpcbenni/trunk/covise/src/application/hlrs/CFX/radialrunner.def

#delete unneeded RadialRunner files
rm -rf radialrunner_*;rm -f rr_meridian_*;rm -f rr_gridparams*;rm -f RadialRunner.deb;rm -f MERIDIANelems*;rm -f MERIDIANnodes_*;rm -f rr_blnodes_*

#delete unneeded CFX files
rm -f radialrunner.def;rm -f radialrunner.def.lck;rm -f radialrunner.def.gtm


ending1=session_template12.pre
ending2=session.pre
concat1=$machineType$ending1
concat2=$machineType$ending2
echo concat1=$concat1
echo concat2=$concat2

#replace $(CFXDIR) in session template file by actual CFXDIR
cfxd=`echo $CFXDIR | sed -e "s/\\\//\\\\\\\\\//g"`

cd $CFXDIR

if [ "$machineType" == "radial" ] || [ "$machineType" == "axial" ] || [ "$machineType" == "rechenraum" ] || [ "$machineType" == "surfacedemo" ] || [ "$machineType" == "plmxml" ]; then
    sed -e "s/\$(CFXDIR)/${cfxd}/" -e "s/REVOLUTIONS/${revolutions}/g" -e "s/MAXITERATIONS/${maxIterations}/g" < $concat1 > $concat2
fi

if [ "$machineType" == "radial_machine" ] || [ "$machineType" == "axial_machine" ] || [ "$machineType" == "complete_machine" ]; then
    sed -e "s/\$(CFXDIR)/${cfxd}/" -e "s/REVOLUTIONS/${revolutions}/g" -e "s/MAXITERATIONS/${maxIterations}/g" -e "s/LOCATION/${locationString}/" < $concat1 > $concat2
fi

cd $CFXDIR/cfx_sub/
touch ghostmesh.gtm
cd $CFXDIR


if [ "$deffile" == "0" ]; then
   echo executing CFX Pre, receiving mesh and bcs from Covise, writing def file 
   if [ "$machineType" == "rechenraum" ]; then
   	echo executing cfx5pre -batch $CFXDIR/cfx12_test.pre
   	cfx5pre -batch $CFXDIR/cfx12_test.pre
  else
	echo executing cfx5pre -batch $CFXDIR/$concat2
	cfx5pre -batch $CFXDIR/$concat2
  fi
   #/mnt/raid/home/hpcmbb/CFX/cfx_sub/user_import
   
   #cfx5pre -session $CFXDIR/$concat2
   
fi

echo starting up solver

cd $CFXDIR/cfx_sub
rm ghostmesh.gtm
cd ~/CFX

echo startup $startup
echo deffile $deffile

if [ "$deffile" == "0" ]; then

  if [ "$startup" == "MPICH Local Parallel"   ]; then
    echo startup is MPICH Local Parallel
    cfxarg=" -part $numProc"
    echo cfx5solve -start-method "$startup" -def $CFXDIR/$machineType.def $cfxarg
    cfx5solve -start-method "$startup" -def $CFXDIR/$machineType.def $cfxarg
  fi
 
  if [ "$startup" == "MPICH Distributed Parallel"   ]; then 
    echo startup is MPICH Distributed Parallel
    cfxarg=" -par-dist $hostlist"
    echo cfx5solve -start-method "$startup" -def $CFXDIR/$machineType.def $cfxarg
    cfx5solve -start-method "$startup" -def $CFXDIR/$machineType.def $cfxarg
  fi

  if [ "$startup" == "serial"  ]; then 
    echo startup is serial
    cfxarg=""
    startup="-serial"
    echo cfx5solve -def $CFXDIR/$machineType.def $cfxarg
    cfx5solve -def $CFXDIR/$machineType.def $cfxarg
  fi

else

  echo cfx5solve -def $deffile $cfxarg
  cfx5solve -def $deffile $cfxarg

fi

#cfx5solve -def complete_machine.def -par-dist "viscluster10*2,viscluster11*2" -start-method "MPICH Distributed Parallel" -part 4 

echo END of Simulation

