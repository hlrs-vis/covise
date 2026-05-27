set(Name "EnergyCore")
set(HEADERS constants.h ButtonBase.h ButtonGroupBase.h ComponentBase.h EditFieldBase.h MenuBase.h SelectionListBase.h Logger.h)
add_subdirectory(interfaces)
add_subdirectory(simulation)
add_subdirectory(utils)

add_library(${Name} INTERFACE ${HEADERS})
target_link_libraries(${Name} INTERFACE core::interface INTERFACE core::simulation INTERFACE core::utils)
COVISE_INSTALL_TARGET(${Name})
