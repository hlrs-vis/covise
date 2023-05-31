SET(LIB_HEADERS
  MultitouchPlugin.h
  MultitouchNavigation.h
)
SET(LIB_SOURCES
  MultitouchPlugin.cpp
  MultitouchNavigation.cpp
)
add_covise_library(CoverMultitouch SHARED ${LIB_HEADERS} ${LIB_SOURCES})
target_link_libraries(CoverMultitouch coOpenCOVER ${OPENSCENEGRAPH_LIBRARIES})
target_compile_definitions(CoverMultitouch PRIVATE COVER_PLUGIN_NAME="CoverMultitouch")
#target_link_libraries(Vrml97Cover coOpenPluginUtil coOpenCOVER coOpenVRUI
#coOSGVRUI ${COVISE_VRBCLIENT_LIBRARY} ${COVISE_CONFIG_LIBRARY}
#${COVISE_UTIL_LIBRARY} ${COVISE_GRMSG_LIBRARY} ${EXTRA_LIBS} ${OPENSCENEGRAPH_LIBRARIES}) # ${CMAKE_THREAD_LIBS_INIT})
covise_install_target(CoverMultitouch)
cover_add_plugin(Multitouch MultitouchPluginPlugin.cpp)
target_link_libraries(Multitouch CoverMultitouch)
#COVER_INSTALL_PLUGIN(Multitouch)
