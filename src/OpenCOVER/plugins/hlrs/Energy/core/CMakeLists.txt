USING(OpenSceneGraph)
set(SOURCES
    PrototypeBuilding.cpp
    TxtInfoboard.cpp
)

set(HEADERS
    interfaces/IBuilding.h
    interfaces/IColorable.h
    interfaces/IDrawable.h
    interfaces/IInfoboard.h
    interfaces/IInformable.h
    interfaces/IMovable.h
    interfaces/ISeperable.h
    interfaces/ITimedependable.h
    PrototypeBuilding.h
    TxtInfoboard.h
)
add_library(EnergyCore OBJECT ${SOURCES} ${HEADERS})
target_link_libraries(EnergyCore PRIVATE ${EXTRA_LIBS})
target_include_directories(EnergyCore PRIVATE ${OPENSCENEGRAPH_INCLUDE_DIRS})
set_target_properties(EnergyCore PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE ON)