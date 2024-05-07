file(GLOB_RECURSE SOURCES "*.cpp" "*.h")
add_library(EnergyInterfaces ${SOURCES})
set_target_properties(EnergyInterfaces PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON)