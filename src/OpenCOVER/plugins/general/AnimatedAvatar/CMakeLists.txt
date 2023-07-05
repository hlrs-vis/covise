using (osgCal)

SET(HEADERS
AnimatedAvatar.h
AnimatedAvatarPlugin.h
ModelProvider/ModelProvider.h
ModelProvider/FBX.h
ModelProvider/Cal3d.h
)

SET(SOURCES
AnimatedAvatar.cpp
AnimatedAvatarPlugin.cpp
ModelProvider/FBX.cpp
ModelProvider/Cal3d.cpp
)

find_package(osgAnimation)

cover_add_plugin(AnimatedAvatar)
target_link_libraries(AnimatedAvatar ${OPENSCENEGRAPH_LIBRARIES} ${OSGANIMATION_LIBRARY})
if(OSGANIMATION_FOUND)
    target_compile_definitions(AnimatedAvatar PRIVATE HAVE_FBX)
endif()
