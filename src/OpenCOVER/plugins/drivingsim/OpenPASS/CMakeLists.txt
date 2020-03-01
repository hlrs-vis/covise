USING(TrafficSimulation OpenPASS)
set (CMAKE_CXX_STANDARD 17)
set(HEADERS
  OpenPASS.h

)
set(SOURCES
  OpenPASS.cpp
)

INCLUDE_DIRECTORIES(
)

cover_add_plugin(OpenPASS ${HEADERS} ${SOURCES})
if(NOT MSVC)
    add_covise_compile_flags(OpenPASS "-Wno-error=deprecated")
endif()
target_link_libraries(OpenPASS ${EXTRA_LIBS})
