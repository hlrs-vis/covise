using(VRML)

SET(HEADERS
    SalzburgFestival.h
    VrmlNodeTangible.h
)

SET(SOURCES
    SalzburgFestival.cpp
    VrmlNodeTangible.cpp
)

cover_add_plugin(TangibleInterface)
TARGET_LINK_LIBRARIES(TangibleInterface coUtil coConfig)
