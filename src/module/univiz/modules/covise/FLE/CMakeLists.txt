SET(HEADERS
  FLE.h
  ${COVISEDIR}/src/module/univiz/libs/unisys/unisys.h
  ${COVISEDIR}/src/module/univiz/libs/unstructured/unstructured.h
  ${COVISEDIR}/src/module/univiz/libs/unifield/unifield.h
  ${COVISEDIR}/src/module/univiz/libs/covise_ext/covise_ext.h
  ${COVISEDIR}/src/module/univiz/libs/unigeom/unigeom.h
  ${COVISEDIR}/src/module/univiz/libs/linalg/linalg.h
)

SET(SOURCES
  FLE.cpp
  ${COVISEDIR}/src/module/univiz/libs/unisys/unisys.cpp
  ${COVISEDIR}/src/module/univiz/libs/unstructured/unstructured.cpp
  ${COVISEDIR}/src/module/univiz/libs/unifield/unifield.cpp
  ${COVISEDIR}/src/module/univiz/libs/covise_ext/covise_ext.cpp
  ${COVISEDIR}/src/module/univiz/libs/unigeom/unigeom.cpp
)

SET(EXTRASOURCES
  FLE.h
  ${COVISEDIR}/src/module/univiz/libs/unisys/unisys.h
  ${COVISEDIR}/src/module/univiz/libs/unstructured/unstructured.h
  ${COVISEDIR}/src/module/univiz/libs/unifield/unifield.h
  ${COVISEDIR}/src/module/univiz/libs/covise_ext/covise_ext.h
  ${COVISEDIR}/src/module/univiz/libs/unigeom/unigeom.h
  ${COVISEDIR}/src/module/univiz/libs/linalg/linalg.h
)
INCLUDE_DIRECTORIES(${COVISEDIR}/src/module/univiz/libs/linalg
${COVISEDIR}/src/module/univiz/libs/unifield
${COVISEDIR}/src/module/univiz/libs/unigeom
${COVISEDIR}/src/module/univiz/libs/unstructured
${COVISEDIR}/src/module/univiz/libs/unisys
${COVISEDIR}/src/module/univiz/libs/covise_ext
${COVISEDIR}/src/module/univiz/modules/impl/FLE
)
ADD_DEFINITIONS(-DCOVISE)

ADD_COVISE_MODULE(Univiz FLE ${EXTRASOURCES} )
covise_wnoerror(FLE)
TARGET_LINK_LIBRARIES(FLE coApi coAppl coCore )

COVISE_INSTALL_TARGET(FLE)
