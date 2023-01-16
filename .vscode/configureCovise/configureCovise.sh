# @args: covise dir, archsuffix, dependencyPath, Qt version, generator, overwrite
echo configuring COVISE with "$@"
export COVISEDIR=$1
export EXTERNLIBS=$3
. $1/.covise.sh

cd $1/.vscode/configureCovise
printenv > covise.env

if [ ! -f build/configureVsCodeSettings ]; then
  rm -r build
  mkdir build
  cd build
  GENERATOR=$5
  cmake -DCMAKE_PREFIX_PATH=$EXTERNLIBS -G $GENERATOR ..
  if [ $GENERATOR = "Ninja" ]; then
    ninja
  elif [ $GENERATOR = "Unix Makefiles" ]; then
    make
  fi
else
cd build
fi
./configureVsCodeSettings "$@"