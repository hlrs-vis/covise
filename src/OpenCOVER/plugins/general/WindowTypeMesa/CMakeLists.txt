USING(OpenGL)

set(HEADERS
    WindowTypeMesa.h
)

set(SOURCES
    WindowTypeMesa.cpp
)

cover_add_plugin(WindowTypeMesa ${HEADERS} ${SOURCES} ${QRC_SOURCES} ${UI_SOURCES})
