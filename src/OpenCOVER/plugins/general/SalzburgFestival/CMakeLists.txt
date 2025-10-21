using(VRML)

SET(HEADERS
    SalzburgFestival.h
)

SET(SOURCES
    SalzburgFestival.cpp
)

cover_add_plugin(SalzburgFestival)
TARGET_LINK_LIBRARIES(SalzburgFestival coUtil coConfig)