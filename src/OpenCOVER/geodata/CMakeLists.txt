using(OpenSceneGraph)
covise_find_package(proj)

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
    coOpenCOVER
    ${COVISE_UTIL_LIBRARY}
)

if(PROJ_FOUND)
    target_include_directories(coVRGeoData PUBLIC ${PROJ_INCLUDE_DIRS})
    target_link_libraries(coVRGeoData ${PROJ_LIBRARIES})
    target_compile_definitions(coVRGeoData PRIVATE -DHAVE_PROJ)
endif()

covise_install_target(coVRGeoData)
covise_install_headers(geodata ${GEODATA_HEADERS})
