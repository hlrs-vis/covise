
SET(HEADERS
   MADIconnect.h
)

SET(SOURCES
   MADIconnect.cpp
)

cover_add_plugin(MADIconnect ${HEADERS} ${SOURCES})

qt_use_modules(MADIconnect Network)

target_link_libraries(MADIconnect ${EXTRA_LIBS})

