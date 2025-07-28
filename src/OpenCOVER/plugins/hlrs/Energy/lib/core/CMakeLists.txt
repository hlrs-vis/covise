USING(OpenSceneGraph)
set(SOURCES
    utils/color.cpp
    utils/osgUtils.cpp
    utils/math.cpp
    simulation/heating.cpp
    simulation/power.cpp
    simulation/simulation.cpp
)

set(HEADERS
    interfaces/IBuilding.h
    interfaces/IEnergyGrid.h
    interfaces/IColorable.h
    interfaces/IDrawable.h
    interfaces/IDrawables.h
    interfaces/IInfoboard.h
    interfaces/IInformable.h
    interfaces/IMovable.h
    interfaces/ISeparable.h
    interfaces/ITimedependable.h
    interfaces/ISolarPanel.h
    utils/color.h
    utils/osgUtils.h
    utils/math.h
    simulation/heating.h
    simulation/simulation.h
    simulation/power.h
    simulation/object.h
    constants.h
)

set(Name "EnergyCore")
add_library(${Name} OBJECT ${SOURCES} ${HEADERS})
target_link_libraries(${Name} ${EXTRA_LIBS})
target_include_directories(${Name} PRIVATE ${OPENSCENEGRAPH_INCLUDE_DIRS})
set_target_properties(${Name} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(${Name} PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON)