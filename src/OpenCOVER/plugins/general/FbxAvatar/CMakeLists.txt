
SET(HEADERS
FbxAvatar.h
InverseKinematic.h
)

SET(SOURCES
FbxAvatar.cpp
InverseKinematic.cpp
)

# find_package(OpenSceneGraph COMPONENTS osgAnimation)

find_package(osgAnimation)
if(NOT OSGANIMATION_FOUND)
    return()
endif()
cover_add_plugin(FbxAvatar ${HEADERS})
target_link_libraries(FbxAvatar ${OPENSCENEGRAPH_LIBRARIES} ${OSGANIMATION_LIBRARY})
