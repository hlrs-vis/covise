SET(LIB_HEADERS
   VRCoviseGeometryManager.h
   SmokeGeneratorSolutions.h
)
SET(LIB_SOURCES
   VRCoviseGeometryManager.cpp
   SmokeGeneratorSolutions.cpp
)
add_covise_library(COVISEPluginUtil SHARED ${LIB_SOURCES})
target_link_libraries(COVISEPluginUtil coOpenPluginUtil coOpenCOVER coOpenVRUI
   coOSGVRUI ${COVISE_APPL_LIBRARY}
   ${COVISE_CORE_LIBRARY} ${COVISE_DO_LIBRARY} ${COVISE_SHM_LIBRARY}
   ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_CONFIG_LIBRARY}
   ${COVISE_UTIL_LIBRARY} ${COVISE_GRMSG_LIBRARY} ${OPENSCENEGRAPH_LIBRARIES}) 
covise_install_target(COVISEPluginUtil)

SET(HEADERS
   CovisePlugin.h
   CoviseRenderObject.h
   CoviseSG.h
   VRCoviseConnection.h
   VRCoviseObjectManager.h
   VRSlider.h
   VRRotator.h
   VRVectorInteractor.h
   coCoviseInteractor.h
   coVRMenuList.h
   coVRTUIParam.h
   coVRDistributionManager.h
   coVRParallelRenderPlugin.h
)

SET(SOURCES
   CovisePlugin.cpp
   CoviseRenderObject.cpp
   CoviseSG.cpp
   VRCoviseConnection.cpp
   VRCoviseObjectManager.cpp
   VRSlider.cpp
   VRRotator.cpp
   VRVectorInteractor.cpp
   coCoviseInteractor.cpp
   coVRMenuList.cpp
   coVRTUIParam.cpp
   coVRDistributionManager.cpp
   coVRParallelRenderPlugin.cpp
)

INCLUDE_DIRECTORIES(
  )

cover_add_plugin(CovisePlugin)
TARGET_LINK_LIBRARIES(CovisePlugin COVISEPluginUtil ${COVISE_APPL_LIBRARY}
   ${COVISE_CORE_LIBRARY} ${COVISE_DO_LIBRARY} ${COVISE_SHM_LIBRARY}
   ${EXTRA_LIBS})
