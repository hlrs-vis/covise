USING (XERCESC)

SET(HEADERS
  UserInterface.h
)

SET(SOURCES
  UserInterface.cpp
)



QT_WRAP_CPP(MOCS ${HEADERS})
set(SOURCES ${SOURCES} ${MOCS})
cover_add_plugin(UserInterface ${HEADERS} ${SOURCES} ${MOCS})


qt_use_modules(UserInterface Core)
