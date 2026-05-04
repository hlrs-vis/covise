covise_find_package(glm)
covise_find_package(proj)

# Workaround for GLM 0.9.9 broken glmTargets.cmake
if(NOT TARGET glm::glm AND GLM_INCLUDE_DIRS)
    add_library(glm::glm INTERFACE IMPORTED)
    set_target_properties(glm::glm PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIRS}"
    )
endif()

# glm: allow .xyz() and similar access
add_definitions(-DGLM_FORCE_SWIZZLE -DGLM_ENABLE_EXPERIMENTAL)

set(GEODATA_HEADERS 
    GeoData.h
)

set(GEODATA_SOURCES 
    GeoData.cpp
)

add_covise_library(coVRGeoData SHARED ${GEODATA_SOURCES} ${GEODATA_HEADERS})

target_link_libraries(coVRGeoData
    # coOpenConfig
    # coConfig
    ${COVISE_UTIL_LIBRARY}
    glm::glm
)

if(PROJ_FOUND)
    target_include_directories(coVRGeoData PUBLIC ${PROJ_INCLUDE_DIRS})
    target_link_libraries(coVRGeoData ${PROJ_LIBRARIES})
    target_compile_definitions(coVRGeoData PRIVATE -DHAVE_PROJ)
endif()

covise_install_target(coVRGeoData)
covise_install_headers(geodata ${GEODATA_HEADERS})
