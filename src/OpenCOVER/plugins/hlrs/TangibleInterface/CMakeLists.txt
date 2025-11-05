using(VRML)

SET(HEADERS
    TangibleInterface.h
    VrmlNodeTangible.h
)

SET(SOURCES
    TangibleInterface.cpp
    VrmlNodeTangible.cpp
)

cover_add_plugin(TangibleInterface)
TARGET_LINK_LIBRARIES(TangibleInterface coUtil coConfig)
