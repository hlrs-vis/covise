USING (XERCESC)

SET(HEADERS
  Beispiel.h
)

SET(SOURCES
  Beispiel.cpp
)



QT_WRAP_CPP(MOCS ${HEADERS})
set(SOURCES ${SOURCES} ${MOCS})
cover_add_plugin(Beispiel ${HEADERS} ${SOURCES} ${MOCS})


qt_use_modules(Beispiel Core)
