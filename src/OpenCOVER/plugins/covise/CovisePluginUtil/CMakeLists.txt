IF(NOT OPENSCENEGRAPH_FOUND)
  RETURN()
ENDIF()

ADD_DEFINITIONS(-DCOVISE_PLUGIN_UTIL)

INCLUDE_DIRECTORIES(
  ${OPENSCENEGRAPH_INCLUDE_DIRS}
)

SET(LIB_HEADERS
   coBaseCoviseInteractor.h
   VRCoviseGeometryManager.h
   SmokeGeneratorSolutions.h
)

SET(LIB_SOURCES
   coBaseCoviseInteractor.cpp
   VRCoviseGeometryManager.cpp
   SmokeGeneratorSolutions.cpp
)

add_covise_library(CovisePluginUtil SHARED ${LIB_SOURCES})
target_link_libraries(CovisePluginUtil coOpenPluginUtil coOpenCOVER coOpenVRUI
   coOSGVRUI ${COVISE_APPL_LIBRARY}
   ${COVISE_CORE_LIBRARY} ${COVISE_DO_LIBRARY} ${COVISE_SHM_LIBRARY}
   ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_CONFIG_LIBRARY}
   ${COVISE_UTIL_LIBRARY} ${COVISE_GRMSG_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES}) 
covise_install_target(CovisePluginUtil)


