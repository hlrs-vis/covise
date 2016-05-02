FIND_PACKAGE(CFX)
IF (NOT CFX_FOUND)
   RETURN()
ENDIF(NOT CFX_FOUND)

INCLUDE_DIRECTORIES(${CFX_INCLUDE_DIRS})


SET(HEADERS
  DumpCFX.h
  $ENV{COVISEDIR}/src/module/univiz/libs/unisys/unisys.h
  $ENV{COVISEDIR}/src/module/univiz/libs/unstructured/unstructured.h
  $ENV{COVISEDIR}/src/module/univiz/libs/unifield/unifield.h
  $ENV{COVISEDIR}/src/module/univiz/libs/covise_ext/covise_ext.h
  $ENV{COVISEDIR}/src/module/univiz/libs/unigeom/unigeom.h
  $ENV{COVISEDIR}/src/module/univiz/libs/linalg/linalg.h
)

SET(SOURCES
  DumpCFX.cpp
  $ENV{COVISEDIR}/src/module/univiz/libs/unisys/unisys.cpp
  $ENV{COVISEDIR}/src/module/univiz/libs/unstructured/unstructured.cpp
  $ENV{COVISEDIR}/src/module/univiz/libs/unifield/unifield.cpp
  $ENV{COVISEDIR}/src/module/univiz/libs/covise_ext/covise_ext.cpp
  $ENV{COVISEDIR}/src/module/univiz/libs/unigeom/unigeom.cpp
)

SET(EXTRASOURCES
  DumpCFX.h
  $ENV{COVISEDIR}/src/module/univiz/libs/unisys/unisys.h
  $ENV{COVISEDIR}/src/module/univiz/libs/unstructured/unstructured.h
  $ENV{COVISEDIR}/src/module/univiz/libs/unifield/unifield.h
  $ENV{COVISEDIR}/src/module/univiz/libs/covise_ext/covise_ext.h
  $ENV{COVISEDIR}/src/module/univiz/libs/unigeom/unigeom.h
  $ENV{COVISEDIR}/src/module/univiz/libs/linalg/linalg.h
)
INCLUDE_DIRECTORIES($ENV{COVISEDIR}/src/module/univiz/libs/linalg
$ENV{COVISEDIR}/src/module/univiz/libs/unstructured
$ENV{COVISEDIR}/src/module/univiz/libs/unisys
$ENV{COVISEDIR}/src/module/univiz/libs/covise_ext
$ENV{COVISEDIR}/src/module/univiz/modules/impl/dump_cfx
)
ADD_DEFINITIONS(-DCOVISE)
ADD_COVISE_MODULE(Univiz DumpCFX ${EXTRASOURCES} )
covise_wnoerror(DumpCFX)
TARGET_LINK_LIBRARIES(DumpCFX coApi coAppl coCore ${CFX_LIBRARIES})

COVISE_INSTALL_TARGET(DumpCFX)
