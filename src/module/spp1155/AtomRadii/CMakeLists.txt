INCLUDE_DIRECTORIES(
../AtomColors
)
SET(HEADERS
  ../AtomColors/AtomColors.h
)
SET(SOURCES
  ../AtomColors/AtomColors.cpp
)
ADD_DEFINITIONS("-DATOMRADII")

ADD_COVISE_MODULE(Tools AtomRadii ${EXTRASOURCES})
qt_use_modules(AtomRadii Xml)
TARGET_LINK_LIBRARIES(AtomRadii coApi coAppl coCore coAlg)
COVISE_INSTALL_TARGET(AtomRadii)
