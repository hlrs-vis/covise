using(VRML)
covise_find_package(OSGEPHEMERIS)

covise_find_package(exiv2)
if(NOT TARGET exiv2lib)
  return()
endif()

set(HEADERS
    SkySphere.h
    VrmlNodeSkySphere.h
)

set(SOURCES
    SkySphere.cpp
    VrmlNodeSkySphere.cpp
)

if(OSGEPHEMERIS_FOUND)
    LIST(APPEND HEADERS EphemeralSky.h)
    LIST(APPEND SOURCES EphemeralSky.cpp)
endif()

cover_add_plugin(SkySphere coVRGeoData exiv2lib)

if(OSGEPHEMERIS_FOUND)
    target_include_directories(SkySphere PRIVATE ${OSGEPHEMERIS_INCLUDE_DIR})
    target_link_libraries(SkySphere ${OSGEPHEMERIS_LIBRARIES})
    target_compile_definitions(SkySphere PRIVATE -DHAVE_EPHEMERIS)
endif()
