if(NOT OSGANIMATION_FOUND)
return()
endif()

cover_add_plugin(GhostAvatar GhostAvatar.cpp GhostAvatar.h BoneParser.cpp BoneParser.h PlanarAvatarControls.h PlanarAvatarControls.cpp GhostAvatarControls.h GhostAvatarControls.cpp)
target_link_libraries(GhostAvatar ${OPENSCENEGRAPH_LIBRARIES})