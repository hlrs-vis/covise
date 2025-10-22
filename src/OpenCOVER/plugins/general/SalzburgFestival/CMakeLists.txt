using(VRML)

SET(HEADERS
    SalzburgFestival.h
    VrmlNodeTangible.h
)

SET(SOURCES
    SalzburgFestival.cpp
    VrmlNodeTangible.cpp
)

cover_add_plugin(SalzburgFestival)
TARGET_LINK_LIBRARIES(SalzburgFestival coUtil coConfig)
