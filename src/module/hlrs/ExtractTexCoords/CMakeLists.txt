SET(HEADERS
)
SET(SOURCES
  ExtractTexCoords.cpp
)
ADD_COVISE_MODULE(IO ExtractTexCoords ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(ExtractTexCoords coApi coAppl coCore coImage)
COVISE_INSTALL_TARGET(ExtractTexCoords)
