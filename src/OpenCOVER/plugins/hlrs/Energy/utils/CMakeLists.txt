file(GLOB_RECURSE SOURCES "*.cpp" "*.h")
add_library(EnergyUtils OBJECT ${SOURCES})
set_target_properties(EnergyUtils PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON)