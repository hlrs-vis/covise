add_subdirectory(core)
add_subdirectory(ennovatis)
add_subdirectory(apache)

set(Name "Energy")
add_library(${Name} STATIC $<TARGET_OBJECTS:EnergyCore> $<TARGET_OBJECTS:Ennovatis> $<TARGET_OBJECTS:Apache_Arrow>)
target_compile_options(${Name} PRIVATE -fPIC)
COVISE_INSTALL_TARGET(${Name})