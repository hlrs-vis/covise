find_package(Boost REQUIRED)
find_package(OpenSceneGraph)

if (NOT OPENSCENEGRAPH_FOUND)
    message(STATUS "Note: not building library virvo_osg (OpenSceneGraph not found)")
    return()
endif()

deskvox_use_package(Boost)
deskvox_use_package(OpenSceneGraph)

deskvox_link_libraries(virvo)
deskvox_link_libraries(virvo_fileio)

deskvox_add_library(virvo_osg
    VolumeDrawable.cpp
)
