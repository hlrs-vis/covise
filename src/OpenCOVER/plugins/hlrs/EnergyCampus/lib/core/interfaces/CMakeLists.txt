USING(OpenSceneGraph)

set(HEADERS
    IBuilding.h
    IEnergyGrid.h
    IColorable.h
    IDrawable.h
    IDrawables.h
    IInfoboard.h
    IInformable.h
    IMovable.h
    ITimedependable.h
    ISolarPanel.h
    ISystem.h
)

set(Name "interface")
add_library(${Name} INTERFACE ${HEADERS})
target_link_libraries(${Name} INTERFACE ${EXTRA_LIBS})
target_include_directories(${Name} INTERFACE ${OPENSCENEGRAPH_INCLUDE_DIRS})
add_library(core::interface ALIAS ${Name})
COVISE_INSTALL_TARGET(${Name})