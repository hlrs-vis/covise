set(HEADERS
    IBuilding.h
    IEnergyGrid.h
    IColorable.h
    IDrawable.h
    IInfoboard.h
    IInformable.h
    ILogger.h
    IMovable.h
    ITimedependable.h
    ISolarPanel.h
    ISystem.h
    IUpdateable.h
)

set(Name "interface")
add_library(${Name} INTERFACE ${HEADERS})
add_library(core::interface ALIAS ${Name})
COVISE_INSTALL_TARGET(${Name})
