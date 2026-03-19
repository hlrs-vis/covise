if(NOT OSGANIMATION_FOUND)
return()
endif()

cover_add_plugin(GhostAvatar GhostAvatar.cpp GhostAvatar.h Bone.cpp Bone.h)
target_link_libraries(GhostAvatar ${OPENSCENEGRAPH_LIBRARIES})