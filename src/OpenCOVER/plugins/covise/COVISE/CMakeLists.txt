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
    ..
  )

cover_add_plugin(CovisePlugin)
target_compile_definitions(CovisePlugin PRIVATE COVER_PLUGIN_NAME="COVISE")
TARGET_LINK_LIBRARIES(CovisePlugin CovisePluginUtil ${COVISE_APPL_LIBRARY}
   ${COVISE_CORE_LIBRARY} ${COVISE_DO_LIBRARY} ${COVISE_SHM_LIBRARY}
   ${EXTRA_LIBS})
set_target_properties(CovisePlugin PROPERTIES OUTPUT_NAME "COVISE")
