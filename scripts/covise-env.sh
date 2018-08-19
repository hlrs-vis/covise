# as this script is sometimes used from shell initialization files,
# do not use exit on errors, but skip to end
unset COENVERROR

if [ -z "$COVISEDIR" ]; then
   echo "COVISEDIR not set (Did you source .covise.sh?)"
   COENVERROR=1
fi

if [ -z "$COENVERROR" ]; then
   # the Inventor renderer does not show any labels and crashes without
   export FL_FONT_PATH="${COVISEDIR}/share/covise/fonts"
   # Coin only has a default font without
   export COIN_FONT_PATH="${COVISEDIR}/share/covise/fonts"

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

   if [ -z "$EXTERNLIBS" ]; then
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

ALVAR_PLUGIN_PATH=${EXTERNLIBS}/alvar/bin/alvarplugins
if [ -d "${ALVAR_PLUGIN_PATH}" ]; then
    export ALVAR_PLUGIN_PATH
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
   case "${ARCHSUFFIX%opt}" in
      linux64|amd64|x64|bishorn|fujisan|monshuygens|lycaeus|maunaloa|gorely|leonidas|constantine|goddard|laughlin|lovelock|verne|rhel3|rhel4|rhel5|rhel51|rhel52|rhel53|rhel6|rhel7|leguan|waran|basilisk|iguana|tuatara|mabuya|drusenkopf|lipinia|slowworm|neolamprologus|saara|julidochromis|indicus|mamba)
         primlibdir=lib64
         ;;
   esac

   ### add our own libraries
   eval export ${libvar}=${empty}:${extLibPath}:${COVISEDIR}/${ARCHSUFFIX}/lib:${EXTERNLIBS}/ALL/${primlibdir}:${EXTERNLIBS}/ALL/lib:\$${libvar}:${COVISEDIR}/${ARCHSUFFIX}/lib/OpenCOVER/plugins

   # Sanity: remove dummy/lib
   eval export ${libvar}=\`echo \$${libvar} \| sed -e 's+:dummy/lib\[0-9\]\*:+:+g'\`

   # Sanity: remove colons at begin+end and double colons in path
   eval export ${libvar}=\`echo \$${libvar} \| sed -e 's+::+:+g' -e 's+^:++g' -e 's+:\$++g'\`

   export PATH="${COVISEDIR}/${ARCHSUFFIX}/bin:$PATH"

   unset libvar

   case "${ARCHSUFFIX}" in
      gcc3|gcc3opt|gcc4|gcc4opt|teck|teckopt|icc|iccopt|insure-gcc3)
      if grep -i -v -q 'redhat\|fedora' /etc/issue; then
         if [ -n "$LD_LIBRARY_PATH" ]; then
            export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${EXTERNLIBS}/rh8-compat/lib"
         else
            export LD_LIBRARY_PATH="${EXTERNLIBS}/rh8-compat/lib"
         fi
      fi
      ;;
      chuckwalla)
      if [ -n "$LD_LIBRARY_PATH" ]; then
         export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${EXTERNLIBS}/ABAQUS/6.5-4/cae/exec/lbr:${EXTERNLIBS}/ABAQUS/6.5-4/cae/External"
      else
         export LD_LIBRARY_PATH="${EXTERNLIBS}/ABAQUS/6.5-4/cae/exec/lbr:${EXTERNLIBS}/ABAQUS/6.5-4/cae/External"
      fi
      ;;
      macx|macxopt|tiger|tigeropt|osx11|osx11opt)
      export DYLD_FRAMEWORK_PATH="${EXTERNLIBS}:${EXTERNLIBS}/OpenSceneGraph:${EXTERNLIBS}/qt/lib"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/Xerces.framework/Versions/Current:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/OpenSceneGraph/lib/osgPlugins:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="/System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources:$DYLD_LIBRARY_PATH"
      ;;
      leopard|leopardopt)
      export DYLD_FRAMEWORK_PATH="${EXTERNLIBS}:${EXTERNLIBS}/OpenSceneGraph:${EXTERNLIBS}/qt/lib"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/xercesc/lib:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/OpenSceneGraph/lib/osgPlugins:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="/System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources:$DYLD_LIBRARY_PATH"
      ;;
      lion|lionopt)
      export DYLD_FRAMEWORK_PATH="${EXTERNLIBS}:${EXTERNLIBS}/openscenegraph:${EXTERNLIBS}/qt4/lib"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/xercesc/lib:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/inventor/lib:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/openscenegraph/lib:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/openscenegraph/lib/osgPlugins:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="/System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources:$DYLD_LIBRARY_PATH"
      ;;
      libc++|libc++opt)
      export DYLD_FRAMEWORK_PATH="${EXTERNLIBS}:${EXTERNLIBS}/openscenegraph:${EXTERNLIBS}/qt5/lib"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/xercesc/lib:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/inventor/lib:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/openscenegraph/lib:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/openscenegraph/lib/osgPlugins:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="/System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources:$DYLD_LIBRARY_PATH"
      ;;
      macos|macosopt)
      export DYLD_FRAMEWORK_PATH="${COVISE_DYLD_FRAMEWORK_PATH}:${DYLD_FRAMEWORK_PATH}"
      export DYLD_LIBRARY_PATH="${COVISE_DYLD_LIBRARY_PATH}:${DYLD_LIBRARY_PATH}"
      export DYLD_FRAMEWORK_PATH="${EXTERNLIBS}:${EXTERNLIBS}/openscenegraph:${EXTERNLIBS}/qt5/lib"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/xercesc/lib:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/inventor/lib:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/openscenegraph/lib:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="${EXTERNLIBS}/openscenegraph/lib/osgPlugins:$DYLD_LIBRARY_PATH"
      export DYLD_LIBRARY_PATH="/System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources:$DYLD_LIBRARY_PATH"
      export COVISE_DYLD_FRAMEWORK_PATH="${DYLD_FRAMEWORK_PATH}"
      export COVISE_DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}"
      ;;
   esac

   # font path for OpenInventor (Type1 and TrueType)
   if [ -z "$OIV_HOME" ]; then
      export OIV_PSFONT_PATH="${OIV_HOME}/data/fonts"
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
