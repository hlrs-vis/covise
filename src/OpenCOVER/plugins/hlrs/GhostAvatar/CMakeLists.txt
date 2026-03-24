if(NOT OSGANIMATION_FOUND)
return()
endif()

cover_add_plugin(GhostAvatar GhostAvatar.cpp GhostAvatar.h BoneParser.cpp BoneParser.h)
target_link_libraries(GhostAvatar ${OPENSCENEGRAPH_LIBRARIES})