if(NOT OSGANIMATION_FOUND)
return()
endif()

cover_add_plugin(OsgAnimation OsgAnimation.cpp)
target_link_libraries(OsgAnimation ${OPENSCENEGRAPH_LIBRARIES})