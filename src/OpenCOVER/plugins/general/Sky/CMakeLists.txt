using(VRML)
covise_find_package(OSGEPHEMERIS)

covise_find_package(exiv2)
if(NOT TARGET exiv2lib)
  return()
endif()

set(HEADERS
    Sky.h
    VrmlNodeSky.h
)

set(SOURCES
    Sky.cpp
    VrmlNodeSky.cpp
)

if(OSGEPHEMERIS_FOUND)
    LIST(APPEND HEADERS EphemeralSky.h)
    LIST(APPEND SOURCES EphemeralSky.cpp)
endif()

cover_add_plugin(Sky coVRGeoData exiv2lib)

if(OSGEPHEMERIS_FOUND)
    target_include_directories(Sky PRIVATE ${OSGEPHEMERIS_INCLUDE_DIR})
    target_link_libraries(Sky ${OSGEPHEMERIS_LIBRARIES})
    target_compile_definitions(Sky PRIVATE -DHAVE_EPHEMERIS)
endif()
