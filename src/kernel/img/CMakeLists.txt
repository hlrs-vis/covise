# @file
# 
# CMakeLists.txt for kernel - img
#
# @author Blasius Czink

using(JPEG)
using(TIFF)
using(PNG)

ADD_DEFINITIONS(-DCOIMAGE_EXPORT)

ADD_DEFINITIONS(${PNG_DEFINITIONS})

SET(IMG_SOURCES
  coBadImage.cpp
  coBinImage.cpp
  coImage.cpp
  coImageImpl.cpp
  coImagePNG.cpp
  coImageTEMPL.cpp
  coImageTIFF.cpp
)

SET(IMG_HEADERS
  coBadImage.h
  coImage.h
  coImagePNG.h
  coImageTEMPL.h
  coBinImage.h
  coImageImpl.h
  coImageRGB.h
  coImageTIFF.h 
)

ADD_COVISE_LIBRARY(coImage ${COVISE_LIB_TYPE} ${IMG_SOURCES} ${IMG_HEADERS})
target_link_libraries(coImage ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(coImage)
COVISE_INSTALL_HEADERS(img ${IMG_HEADERS})
