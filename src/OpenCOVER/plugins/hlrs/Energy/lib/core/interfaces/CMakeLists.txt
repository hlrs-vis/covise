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
    ui/IButton.h
    ui/IButtonGroup.h
    ui/IEditField.h
    ui/IGUIFactory.h
    ui/IMenu.h
)

set(Name "interface")
add_library(${Name} INTERFACE ${HEADERS})
add_library(core::interface ALIAS ${Name})
COVISE_INSTALL_TARGET(${Name})
