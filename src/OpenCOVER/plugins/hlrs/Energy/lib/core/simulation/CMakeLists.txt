set(SOURCES
    heatingresult.cpp
    powerresult.cpp
    scalarpropertiesprocessor.cpp
    collector.cpp
)

set(HEADERS
    heatingresult.h
    unitmap.h
    datastorage.h
    scalarproperties.h
    scalarpropertiesprocessor.h
    collector.h
    simulationresult.h
    powerresult.h
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
