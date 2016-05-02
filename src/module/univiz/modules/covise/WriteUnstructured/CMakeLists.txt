SET(HEADERS
  WriteUnstructured.h
  ${COVISEDIR}/src/module/univiz/libs/unisys/unisys.h
  ${COVISEDIR}/src/module/univiz/libs/unstructured/unstructured.h
  ${COVISEDIR}/src/module/univiz/libs/unifield/unifield.h
  ${COVISEDIR}/src/module/univiz/libs/covise_ext/covise_ext.h
  ${COVISEDIR}/src/module/univiz/libs/unigeom/unigeom.h
  ${COVISEDIR}/src/module/univiz/libs/linalg/linalg.h
)

SET(SOURCES
  WriteUnstructured.cpp
  ${COVISEDIR}/src/module/univiz/libs/unisys/unisys.cpp
  ${COVISEDIR}/src/module/univiz/libs/unstructured/unstructured.cpp
  ${COVISEDIR}/src/module/univiz/libs/unifield/unifield.cpp
  ${COVISEDIR}/src/module/univiz/libs/covise_ext/covise_ext.cpp
  ${COVISEDIR}/src/module/univiz/libs/unigeom/unigeom.cpp
)

SET(EXTRASOURCES
  WriteUnstructured.h
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
${COVISEDIR}/src/module/univiz/modules/impl/write_unstructured
)
ADD_DEFINITIONS(-DCOVISE)

ADD_COVISE_MODULE(Univiz WriteUnstructured ${EXTRASOURCES} )
covise_wnoerror(WriteUnstructured)
TARGET_LINK_LIBRARIES(WriteUnstructured coApi coAppl coCore )

COVISE_INSTALL_TARGET(WriteUnstructured)
