using(TIFF)
using(JPEG)
using(PNG)

ADD_DEFINITIONS(

)

SET(HEADERS
  
)

SET(SOURCES
  BBoxes.cpp
  ImageToTexture.cpp
  TextureMapping.cpp
)

SET(EXTRASOURCES
  BBoxes.h
  ImageToTexture.h
  TextureMapping.h
)

ADD_COVISE_MODULE(Tools ImageToTexture ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(ImageToTexture  coApi coAppl coCore coUtil ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(ImageToTexture)
