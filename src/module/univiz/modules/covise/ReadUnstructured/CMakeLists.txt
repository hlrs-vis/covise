SET(HEADERS
  ReadUnstructured.h
  ${COVISEDIR}/src/module/univiz/libs/unisys/unisys.h
  ${COVISEDIR}/src/module/univiz/libs/unstructured/unstructured.h
  ${COVISEDIR}/src/module/univiz/libs/unifield/unifield.h
  ${COVISEDIR}/src/module/univiz/libs/covise_ext/covise_ext.h
  ${COVISEDIR}/src/module/univiz/libs/unigeom/unigeom.h
  ${COVISEDIR}/src/module/univiz/libs/linalg/linalg.h
)

SET(SOURCES
  ReadUnstructured.cpp
  ${COVISEDIR}/src/module/univiz/libs/unisys/unisys.cpp
  ${COVISEDIR}/src/module/univiz/libs/unstructured/unstructured.cpp
  ${COVISEDIR}/src/module/univiz/libs/unifield/unifield.cpp
  ${COVISEDIR}/src/module/univiz/libs/covise_ext/covise_ext.cpp
  ${COVISEDIR}/src/module/univiz/libs/unigeom/unigeom.cpp
)

SET(EXTRASOURCES
  ReadUnstructured.h
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
)
ADD_DEFINITIONS(-DCOVISE)

ADD_COVISE_MODULE(Univiz ReadUnstructured ${EXTRASOURCES} )
covise_wnoerror(ReadUnstructured)
TARGET_LINK_LIBRARIES(ReadUnstructured coApi coAppl coCore )

COVISE_INSTALL_TARGET(ReadUnstructured)
