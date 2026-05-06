USING(VRML)

covise_find_package(exiv2)
if(NOT TARGET exiv2lib)
  return()
endif()


SET(HEADERS
    SkySphere.h
    VrmlNodeSkySphere.h
)

SET(SOURCES
    SkySphere.cpp
    VrmlNodeSkySphere.cpp
)

cover_add_plugin(SkySphere coVRGeoData exiv2lib)
