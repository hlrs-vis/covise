# as this script is sometimes used from shell initialization files,
# do not use exit on errors, but skip to end
unset COENVERROR

if [ -z "$COVISEDIR" ]; then
   echo "COVISEDIR not set (Did you source .covise.sh?)"
   COENVERROR=1
fi

if [ -z "$COENVERROR" ]; then

   # fonts for OpenSceneGraph
   if [ -z "$OSGFILEPATH" ]; then
      export OSGFILEPATH="${COVISEDIR}/share/covise/fonts:${OSGFILEPATH}"
   else
      export OSGFILEPATH="${COVISEDIR}/share/covise/fonts"
   fi


   if [ -r "$COVISEDIR/scripts/covise-functions.sh" ]; then
      . "${COVISEDIR}/scripts/covise-functions.sh"
   else
      echo "${COVISEDIR}/scripts/covise-functions.sh not found"
      COENVERROR=1
   fi
fi

if [ -z "$COENVERROR" ]; then
   if [ -z "$ARCHSUFFIX" ]; then
       guess_archsuffix
   fi

   if [ -z "$ARCHSUFFIX" ]; then
      echo "ARCHSUFFIX is not set"
      COENVERROR=1
   fi
fi

if [ -z "$COENVERROR" ]; then

   if [ -z "$EXTERNLIBS" -a -d "$EXTERNLIBS" ]; then
      export EXTERNLIBS=`sh -c "cd $COVISEDIR/extern_libs/$ARCHSUFFIX ; pwd -P"`
   fi

   if [ -z "$COVISE_PATH" ]; then
      if [ -z "$COVISEDESTDIR" ]; then
          export COVISE_PATH="$COVISEDIR"
      else
          export COVISE_PATH="$COVISEDESTDIR:$COVISEDIR"
      fi
   fi
fi

if [ -d "$EXTERNLIBS" ]; then
    ALVAR_PLUGIN_PATH=${EXTERNLIBS}/alvar/bin/alvarplugins
    if [ -d "${ALVAR_PLUGIN_PATH}" ]; then
        export ALVAR_PLUGIN_PATH
    fi
fi

if [ -z "$COENVERROR" ]; then
   ### Collect all library pathes of external libs we use
   extLibPath=""
   if [ -d "${EXTERNLIBS}/system/lib" ]; then
      extLibPath="${EXTERNLIBS}/system/lib"
   fi
   if [ -d "${COVISEDIR}/${ARCHSUFFIX}/system-lib" ]; then
      extLibPath="${COVISEDIR}/${ARCHSUFFIX}/system-lib:${extLibPath}"
   fi

   if [ -n "$PYTHONPATH" ]; then
      export PYTHONPATH="${COVISEDIR}/${ARCHSUFFIX}/lib:${PYTHONPATH}"
   else
      export PYTHONPATH="${COVISEDIR}/${ARCHSUFFIX}/lib"
   fi

   ### depending on the platform, the library search path may be in different directories
   if [ "$(uname)" = "Darwin" ]; then
      libvar=DYLD_LIBRARY_PATH
   else
      libvar=LD_LIBRARY_PATH
   fi

   primlibdir=lib
   scndlibdir=lib64
   case "${ARCHSUFFIX%opt}" in
      linux64|amd64|x64|bishorn|fujisan|monshuygens|lycaeus|maunaloa|gorely|leonidas|constantine|goddard|laughlin|lovelock|verne|rhel3|rhel4|rhel5|rhel51|rhel52|rhel53|rhel6|rhel7|leguan|waran|basilisk|iguana|tuatara|mabuya|drusenkopf|lipinia|slowworm|neolamprologus|saara|julidochromis|indicus|mamba|cyprichromis|tangachromis|altolamprologus|leap153|leap154|chalinochromis)
         primlibdir=lib64
         scndlibdir=lib
         ;;
   esac

   ### add our own libraries
   if [ -d "$EXTERNLIBS" ]; then
       eval export ${libvar}=${empty}:${extLibPath}:${COVISEDIR}/${ARCHSUFFIX}/lib:${EXTERNLIBS}/ALL/${primlibdir}:${EXTERNLIBS}/ALL/${scndlibdir}:\$${libvar}:${COVISEDIR}/${ARCHSUFFIX}/lib/OpenCOVER/plugins
   else
       eval export ${libvar}=${empty}:${extLibPath}:${COVISEDIR}/${ARCHSUFFIX}/lib:\$${libvar}:${COVISEDIR}/${ARCHSUFFIX}/lib/OpenCOVER/plugins
   fi

   # Sanity: remove dummy/lib
   eval export ${libvar}=\`echo \$${libvar} \| sed -e 's+:dummy/lib\[0-9\]\*:+:+g'\`

   # Sanity: remove colons at begin+end and double colons in path
   eval export ${libvar}=\`echo \$${libvar} \| sed -e 's+::+:+g' -e 's+^:++g' -e 's+:\$++g'\`

   export PATH="${COVISEDIR}/${ARCHSUFFIX}/bin:$PATH"

   unset libvar

   case "${ARCHSUFFIX}" in
      macos|macosopt)
      export DYLD_FRAMEWORK_PATH="${EXTERNLIBS}/ALL"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/ALL/lib/osgPlugins:$DYLD_LIBRARY_PATH"
      ;;
      spack*)
      if [ "$(uname)" = "Darwin" ]; then
          [ -n "${SPACK_DYLD_FALLBACK_FRAMEWORK_PATH}" ] && export DYLD_FALLBACK_FRAMEWORK_PATH="${SPACK_DYLD_FALLBACK_FRAMEWORK_PATH}"
          [ -n "${SPACK_DYLD_FALLBACK_LIBRARY_PATH}" ] && export DYLD_FALLBACK_LIBRARY_PATH="${SPACK_DYLD_FALLBACK_LIBRARY_PATH}"
      fi
      ;;
   esac
   if [ "$(uname)" = "Darwin" ]; then
      export COVISE_DYLD_FRAMEWORK_PATH="${DYLD_FRAMEWORK_PATH}"
      export COVISE_DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}"
      export COVISE_DYLD_FALLBACK_FRAMEWORK_PATH="${DYLD_FALLBACK_FRAMEWORK_PATH}"
      export COVISE_DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH}"
   fi

   #version 14 no CFX5_UNITS_DIR will be set
   cfxVersion=`which cfx5solve 2> /dev/null` 
   if [[ "$cfxVersion" != *14* ]] ; then
     # unit path for ReadCFX (path to units.cfx)
     export CFX5_UNITS_DIR="${COVISEDIR}/share/covise/cfx"
     #echo "CFX5_UNITS_DIR=${CFX5_UNITS_DIR}"
   else
     unset CFX5_UNITS_DIR
   fi
fi

unset COENVERROR
