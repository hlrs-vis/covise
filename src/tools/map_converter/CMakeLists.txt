# PROJECT(map_converter)

ADD_DEFINITIONS(
)

INCLUDE_DIRECTORIES(
)

SET(HEADERS
  Connection.h
  Environment.h
  ModuleSkel.h
  NetFile.h
  PyFile.h
  Skeletons.h
  Translations.h
)

SET(SOURCES
  Connection.cpp
  Environment.cpp
  MapConverter.cpp
  ModuleSkel.cpp
  NetFile.cpp
  PyFile.cpp
  Skeletons.cpp
  Translations.cpp
)

SET(EXTRASOURCES
  $$(COVISEDIR)/share/covise/map_converster_translations.txt
)

ADD_COVISE_EXECUTABLE(map_converter)

TARGET_LINK_LIBRARIES(map_converter
  coUtil
  coConfig)

qt_use_modules(map_converter Core)

COVISE_INSTALL_TARGET(map_converter)
