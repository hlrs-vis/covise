set(SOURCES
    heating.cpp
    power.cpp
    simulation.cpp
)

set(HEADERS
    heating.h
    simulation.h
    power.h
    object.h
    object_factory.h
    object_type.h
)

set(Name "simulation")
add_library(${Name} STATIC ${SOURCES} ${HEADERS})
add_library(core::simulation ALIAS ${Name})
target_compile_options(${Name} PRIVATE -fPIC)
target_include_directories(${Name} PRIVATE
	"${CMAKE_CURRENT_BINARY_DIR}/include")
COVISE_INSTALL_TARGET(${Name})