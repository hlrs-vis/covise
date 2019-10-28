if (UNIVIZ_VISTLE)
    add_definitions(-DVISTLE)
    add_subdirectory(libs/vistle_ext)
    add_subdirectory(modules/vistle)
endif()

if (UNIVIZ_COVISE)
    add_definitions(-DCOVISE)
    add_subdirectory(modules/covise)
endif()

