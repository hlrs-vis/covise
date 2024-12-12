USING(OpenSceneGraph)
set(SOURCES
    utils/color.cpp
    utils/osgUtils.cpp
    PrototypeBuilding.cpp
    CityGMLBuilding.cpp
    TxtInfoboard.cpp
    EnergyGrid.cpp
    grid.cpp
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
    utils/color.h
    utils/osgUtils.h
    PrototypeBuilding.h
    CityGMLBuilding.h
    TxtInfoboard.h
    EnergyGrid.h
    grid.h
)
ADD_COVISE_LIBRARY(EnergyCore STATIC ${SOURCES} ${HEADERS})
target_link_libraries(EnergyCore ${EXTRA_LIBS})
target_include_directories(EnergyCore PRIVATE ${OPENSCENEGRAPH_INCLUDE_DIRS})
set_target_properties(EnergyCore PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON CXX_STANDARD 20)

COVISE_INSTALL_TARGET(EnergyCore)