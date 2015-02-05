##############################################
#
# COVISE Basic Settings: DO NOT CHANGE
#
################################################

#
# COVISE Installation Directory
#
# If the default (~/covise) is not used, and your cwd is not the covise
# home (i. e. a directory which contains a .covise.sh file), you must set it
# to your covise directory
#

if [ -z "$COVISEDIR" ]; then
   if [ -f .covise.sh ]; then
      export COVISEDIR="`/bin/pwd`"
   elif [ -f "${HOME}"/covise/.covise.sh ]; then
      export COVISEDIR="`sh -f -c 'cd  ~/covise ; unset PWD; /bin/pwd'`"
   elif [ -f "${HOME}"/covise_snap/.covise.sh ]; then
      export COVISEDIR="`sh -f -c 'cd  ~/covise_snap ; unset PWD; /bin/pwd'`"
   else
      echo "Cannot determine COVISEDIR, set manually to directory where bin and config directories reside"
      return 1
   fi
fi

export COVISEDIR="`sh -f -c 'cd "$COVISEDIR" ; unset PWD; /bin/pwd'`"

if [ ! -z "$COVISEDIR" ]; then
    export COVISEDESTDIR=$COVISEDIR
fi

if [ -r "$COVISEDIR"/scripts/covise-functions.sh ]; then
    . "$COVISEDIR"/scripts/covise-functions.sh
else
   echo "$COVISEDIR/scripts/covise-functions.sh not readable"
   return 1
fi

#
# System Architecture
#

guess_archsuffix

# PATH
#
if [ "$COVISE_GLOBALINSTDIR" = "" ]; then
    export PATH="$COVISEDIR"/bin:"$PATH"
    if [ "$COVISE_PATH" = "" ]; then
        export COVISE_PATH="${COVISEDIR}"
        if [ "$COVISEDESTDIR" != "$COVISEDIR" ]; then
            export COVISE_PATH="${COVISEDESTDIR}:${COVISE_PATH}"
        fi
    fi
else
    export PATH="$COVISE_GLOBALINSTDIR"/covise/bin:"$PATH"
    if [ "$COVISE_PATH" = "" ]; then
        export COVISE_PATH="$COVISEDIR":"$COVISE_GLOBALINSTDIR/covise"
    fi
fi

#
# COVISE environment path (colon separated list)
#

BASEARCH=`echo $ARCHSUFFIX | sed -e 's/opt$//' -e 's/mpi$//' -e 's/xenomai$//'  `
if [ -z "$EXTERNLIBS" ]; then
  export EXTERNLIBS="${COVISEDIR}/extern_libs/${ARCHSUFFIX}"
  if [ ! -d "$EXTERNLIBS" ]; then
     export EXTERNLIBS="${COVISEDIR}/extern_libs/${BASEARCH}"
  fi

  if [ ! -d "$EXTERNLIBS" ]; then
     export EXTERNLIBS="/data/extern_libs/$ARCHSUFFIX"
     if [ ! -d "$EXTERNLIBS" ]; then
        export EXTERNLIBS="/data/extern_libs/$BASEARCH"
     fi
  fi
fi

GITDATE=$(GIT_DIR=${COVISEDIR}/.git git log -n1 '--format=%ci')
if [ $? ]; then
   export COVISE_VERSION=$(echo $GITDATE | sed -e 's/-/./' -e 's/-.*//' -e 's/\.0/./')
fi

if [ -z "${PYTHON_HOME}" -a -d "${EXTERNLIBS}/python" ]; then
   export PYTHON_HOME="${EXTERNLIBS}/python"
fi

if [ -r "${HOME}"/.covise.local.sh ]; then 
   . "${HOME}"/.covise.local.sh
fi
