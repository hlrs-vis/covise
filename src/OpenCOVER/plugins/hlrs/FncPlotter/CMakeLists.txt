set(HEADERS
    coPlotCoordSystem.h
)
set(SOURCES
    coPlotCoordSystem.cpp coPlotGraph.cpp
)
cover_add_plugin(FncPlotter)
covise_wnoerror(FncPlotter)
target_link_libraries(FncPlotter)
