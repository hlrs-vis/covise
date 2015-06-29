# - Find ABAQUS
# Find the ABAQUS includes and library
#
#  ABAQUS_INCLUDE_DIR - Where to find ABAQUS includes
#  ABAQUS_LIBRARIES   - List of libraries when using ABAQUS
#  ABAQUS_FOUND       - True if ABAQUS was found
#/data/extern_libs/rhel6/abaqus/lib/libABQSMAOdbApi.so /data/extern_libs/rhel6/abaqus/lib/libABQSMAOdb*.so /data/extern_libs/rhel6/abaqus/lib/libABQSMABasShared.so  /data/extern_libs/rhel6/abaqus/lib/libABQSMAAbuGeom.so -lxerces-c -Wl,-rpath,/mnt/raid/svn/wcs/trunk/covise/rhel6/lib:/data/extern_libs/rhel6/qt5/lib:/data/extern_libs/rhel6/abaqus/lib /data/extern_libs/rhel6/abaqus/lib/libABQSMAAbuBasicUtils.so  /data/extern_libs/rhel6/abaqus/lib/libABQSMAShpCore.so /data/extern_libs/rhel6/abaqus/lib/libABQSMARomDiagEx.so /data/extern_libs/rhel6/abaqus/lib/libABQSMAFeoModules.so /data/extern_libs/rhel6/abaqus/lib/libABQSMAAspSupport.so /data/extern_libs/rhel6/abaqus/lib/libABQSMABlaModule.so /data/extern_libs/rhel6/abaqus/lib/libacml.so  /data/extern_libs/rhel6/abaqus/lib/libifcore.so.5 /data/extern_libs/rhel6/abaqus/lib/libimf.so /data/extern_libs/rhel6/abaqus/lib/libiomp5.so /data/extern_libs/rhel6/abaqus/lib/libmpi.so /data/extern_libs/rhel6/abaqus/lib/libmpio.so


IF(ABAQUS_INCLUDE_DIR)
  SET(ABAQUS_FIND_QUIETLY TRUE)
ENDIF(ABAQUS_INCLUDE_DIR)

set(LIBSEARCH
  $ENV{EXTERNLIBS}/abaqus
  ~/Library/Frameworks
  /Library/Frameworks
  /usr/local
  /usr
  /sw
  /opt/local
  /opt/csw
  /opt
  )

FIND_PATH(ABAQUS_INCLUDE_DIR "odb_Enum.h"
  PATHS
  $ENV{EXTERNLIBS}/abaqus/include
  ~/Library/Frameworks/include
  /Library/Frameworks/include
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
  DOC "ABAQUS - Headers"
)

SET(ABAQUS_NAMES libABQSMAOdbApi.so ABQSMAOdbApi.lib ABQADB_Core)
SET(ABAQUS_DBG_NAMES ABQADB_CoreD)

FIND_LIBRARY(ABAQUS_LIBRARY NAMES ${ABAQUS_NAMES}
  PATHS ${LIBSEARCH}
  PATH_SUFFIXES lib lib64
  DOC "ABAQUS - Library"
)
FIND_LIBRARY(ABAQUS_LIBRARY_2 NAMES libifcore.so.5 ABQSMAAbuBasicUtils.lib
  PATHS ${LIBSEARCH}
  PATH_SUFFIXES lib lib64
  DOC "ABAQUS - Library"
)

FIND_LIBRARY(ABAQUS_LIBRARY_3 NAMES libimf.so ABQSMABasCoreUtils.lib
  PATHS ${LIBSEARCH}
  PATH_SUFFIXES lib lib64
  DOC "ABAQUS - Library"
)

FIND_LIBRARY(ABAQUS_LIBRARY_4 NAMES libiomp5.so ABQSMABasShared.lib
  PATHS ${LIBSEARCH}
  PATH_SUFFIXES lib lib64
  DOC "ABAQUS - Library"
)

FIND_LIBRARY(ABAQUS_LIBRARY_5 NAMES libmpi.so ABQSMAOdbCore.lib
  PATHS ${LIBSEARCH}
  PATH_SUFFIXES lib lib64
  DOC "ABAQUS - Library"
)

FIND_LIBRARY(ABAQUS_LIBRARY_6 NAMES libmpio.so ABQSMABasMem.lib
  PATHS ${LIBSEARCH}
  PATH_SUFFIXES lib lib64
  DOC "ABAQUS - Library"
)

FIND_LIBRARY(ABAQUS_LIBRARY_7 NAMES libABQSMAOdbCore.so
  PATHS ${LIBSEARCH}
  PATH_SUFFIXES lib lib64
  DOC "ABAQUS - Library"
)

FIND_LIBRARY(ABAQUS_LIBRARY_8 NAMES libABQSMAAbuBasicUtils.so
  PATHS ${LIBSEARCH}
  PATH_SUFFIXES lib lib64
  DOC "ABAQUS - Library"
)

FIND_LIBRARY(ABAQUS_LIBRARY_9 NAMES libABQSMABasShared.so
  PATHS ${LIBSEARCH}
  PATH_SUFFIXES lib lib64
  DOC "ABAQUS - Library"
)

FIND_LIBRARY(ABAQUS_LIBRARY_10 NAMES libABQSMABasCoreUtils.so
  PATHS ${LIBSEARCH}
  PATH_SUFFIXES lib lib64
  DOC "ABAQUS - Library"
)

FIND_LIBRARY(ABAQUS_LIBRARY_11 NAMES libABQSMABasMem.so
  PATHS ${LIBSEARCH}
  PATH_SUFFIXES lib lib64
  DOC "ABAQUS - Library"
)

 #abaqus13   libJS0HTTP.so
set (ABAQUS_SEARCH_LIBS
   dl
   pthread
   libABQSMAOdbAttrEO.so
   libABQSMAOdbCoreGeom.so
   libABQSMAShpCore.so
   libABQSMAAspDiagExtractor.so
   libABQSMARomDiagEx.so
   libABQSMARfmInterface.so
   libABQSMAAbuGeom.so
   libABQSMASimInterface.so
   libABQSMABasCatia.so
   libirc.so
   libintlc.so.5
   libABQSMAAspSupport.so
   libABQSMABasGenericsLib.so
   libJS0GROUP.so
   libCATSysMultiThreading.so
   libCATSysTS.so
   libJS0ZLIB.so
   libJS0BASEILB.so
   libABQSMABasPrfTrkLib.so
   libsvml.so
   libifport.so.5
   libifcoremt.so.5
   libABQSMASrvBasic.so
   libABQSMAUzlZlib.so
   libABQSMASimManifestSubcomp.so
   libABQSMASimBulkAPI.so
   libABQSMASimPoolManager.so
   libABQSMASimContainers.so
   libABQSMABasXmlDocument.so
   libABQSMABasXmlParser.so
   libABQSMAAspCommunications.so
   libABQSMAFeoModules.so
   libABQSMAElkCore.so
   libABQDMP_Core.so
   libABQSMAAspSchemaSupport.so
   libABQSMAMtxCoreModule.so
   libABQSMAObjSimObjectsMod.so
   libABQSMAAbuLicense.so
   libABQSMAEliLicense.so
   libCATSysAllocator.so
   libCATMainwin.so
   libABQSMASglSimXmlFndLib.so
   libABQSMAMsgModules.so
   libABQSMABlaModule.so
   libABQSMAFsmShared.so
   libABQSMAMsgCommModules.so
   libABQSMASglSharedLib.so
   libSMAFeaBackbone.so
   libmkl_sequential.so
   libmkl_intel_lp64.so
   libmkl_core.so
   libmkl_intel_thread.so
   libCATSysCommunication.so
   )


INCLUDE(FindPackageHandleStandardArgs)

IF(MSVC)
  # VisualStudio needs a debug version
  FIND_LIBRARY(ABAQUS_LIBRARY_DEBUG NAMES ABQSMAOdbApi.lib
    PATHS
    $ENV{EXTERNLIBS}/abaqus/lib
    DOC "ABAQUS - Library (Debug)"
  )
  FIND_LIBRARY(ABAQUS_LIBRARY_DEBUG_2 NAMES ABQSMAAbuBasicUtils.lib
    PATHS
    $ENV{EXTERNLIBS}/abaqus/lib
    DOC "ABAQUS - Library (Debug)"
  )
  FIND_LIBRARY(ABAQUS_LIBRARY_DEBUG_3 NAMES ABQSMABasCoreUtils.lib
    PATHS
    $ENV{EXTERNLIBS}/abaqus/lib
    DOC "ABAQUS - Library (Debug)"
  )
  FIND_LIBRARY(ABAQUS_LIBRARY_DEBUG_4 NAMES ABQUTI_BLAS_Core ABQSMABasShared.lib
    PATHS
    $ENV{EXTERNLIBS}/abaqus/lib
    DOC "ABAQUS - Library (Debug)"
  )
  FIND_LIBRARY(ABAQUS_LIBRARY_DEBUG_5 NAMES ABQUTI_BasicUtils ABQSMAOdbCore.lib
    PATHS
    $ENV{EXTERNLIBS}/abaqus/lib
    DOC "ABAQUS - Library (Debug)"
  )
  FIND_LIBRARY(ABAQUS_LIBRARY_DEBUG_6 NAMES ABQDDB_Odb ABQSMABasMem.lib
    PATHS
    $ENV{EXTERNLIBS}/abaqus/lib
    DOC "ABAQUS - Library (Debug)"
  )
  
  IF(ABAQUS_LIBRARY_DEBUG AND ABAQUS_LIBRARY)
  
    SET(ABAQUS_LIBRARIES optimized ${ABAQUS_LIBRARY} optimized ${ABAQUS_LIBRARY_2} optimized ${ABAQUS_LIBRARY_3} optimized ${ABAQUS_LIBRARY_4} optimized ${ABAQUS_LIBRARY_5} optimized ${ABAQUS_LIBRARY_6})
    SET(ABAQUS_LIBRARIES ${ABAQUS_LIBRARIES} debug ${ABAQUS_LIBRARY_DEBUG} debug ${ABAQUS_LIBRARY_DEBUG_2} debug ${ABAQUS_LIBRARY_DEBUG_3} debug ${ABAQUS_LIBRARY_DEBUG_4} debug ${ABAQUS_LIBRARY_DEBUG_5} debug ${ABAQUS_LIBRARY_DEBUG_6})
  
  ENDIF(ABAQUS_LIBRARY_DEBUG AND ABAQUS_LIBRARY)

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(ABAQUS DEFAULT_MSG ABAQUS_LIBRARY ABAQUS_LIBRARY_DEBUG ABAQUS_INCLUDE_DIR)

  MARK_AS_ADVANCED(ABAQUS_LIBRARY ABAQUS_LIBRARY_DEBUG ABAQUS_INCLUDE_DIR)
  
ELSE(MSVC)
  # rest of the world
  set(ABAQUS_FOUND_LIBS)
  foreach(lib ${ABAQUS_SEARCH_LIBS})
     unset(ABAQUS_SOME_LIB CACHE)
     find_library(ABAQUS_SOME_LIB NAMES ${lib}
        PATHS ${LIBSEARCH}
        PATH_SUFFIXES lib lib64
        DOC "ABAQUS - Library"
     )
     if (ABAQUS_SOME_LIB)
        #message("abaqus: found ${ABAQUS_SOME_LIB}")
        set(ABAQUS_FOUND_LIBS ${ABAQUS_FOUND_LIBS} ${ABAQUS_SOME_LIB})
     endif()
  endforeach()

  SET(ABAQUS_LIBRARIES ${ABAQUS_LIBRARY} ${ABAQUS_LIBRARY_1}
     ${ABAQUS_LIBRARY_2} ${ABAQUS_LIBRARY_3} ${ABAQUS_LIBRARY_4}
     ${ABAQUS_LIBRARY_5}  ${ABAQUS_LIBRARY_6} ${ABAQUS_LIBRARY_7}
     ${ABAQUS_LIBRARY_8} ${ABAQUS_LIBRARY_9} ${ABAQUS_LIBRARY_10}
     ${ABAQUS_LIBRARY_11}
     ${ABAQUS_FOUND_LIBS}
     )

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(ABAQUS DEFAULT_MSG ABAQUS_LIBRARY ABAQUS_INCLUDE_DIR)
  
  MARK_AS_ADVANCED(ABAQUS_LIBRARY ABAQUS_INCLUDE_DIR)
  
ENDIF(MSVC)

IF(ABAQUS_FOUND)
  SET(ABAQUS_INCLUDE_DIRS ${ABAQUS_INCLUDE_DIR})
ENDIF(ABAQUS_FOUND)
