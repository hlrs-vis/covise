
SET(HEADERS
FbxAvatar.h
InverseKinematic.h
)

SET(SOURCES
FbxAvatar.cpp
InverseKinematic.cpp
)

cover_add_plugin(FbxAvatar ${HEADERS})
target_link_libraries(FbxAvatar ${OPENSCENEGRAPH_LIBRARIES})
