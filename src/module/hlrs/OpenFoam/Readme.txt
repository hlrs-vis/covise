HLRS VisCluster OpenFoam and FunctionObject Installation:

In your home directory:
  mkdir OpenFOAM
  cd OpenFOAM
  mkdir $USER-1.6-ext

Link or copy "OpenFOAM-src" in OpenFoam module directory to "$HOME/OpenFOAM/$USER-1.6-ext" and name it "src":
  ln -s $COVISEDIR/src/application/hlrs/OpenFoam/OpenFOAM-src $HOME/OpenFOAM/$USER-1.6-ext/src

Add to .bashrc:
  export FOAM_INST_DIR=/mnt/raid/soft/OpenFOAM
  source $FOAM_INST_DIR/OpenFOAM-1.6-ext/etc/bashrc

Start a new shell

Enter OpenFoam user source directory and compile solver as well as the function object:
  cd $HOME/OpenFOAM/$USER-1.6-ext/src/simpleFoamCovise
  wmake
  cd $HOME/OpenFOAM/$USER-1.6-ext/src/stateToCovise
  wmake libso


Gate case:

In general, OpenFoam cases are run in the $FOAM_RUN ("$HOME/OpenFOAM/$USER-1.6-ext/run") directory.

A test case from "raid" can be copied there:
  cp -r /mnt/raid/data/OpenFOAM/gate $FOAM_RUN

In Covise, load "covise_gate3.net" (don't forget to compile the OpenFoam
module)

In the OpenFoam module parameters, set the right path to the startup script
($FOAM_RUN/gate.sh)

Execute the Gate module





Needed files in case-directory:

system/controlDict
system/fvSchemes
system/fvSolutions
constant/transportProperties
constant/RASProperties
