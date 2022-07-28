using(FFMPEG)

SET(HEADERS ffmpegEncoder.h)
SET(SOURCES ffmpegEncoder.cpp)
covise_add_library(coFFmpegEncoder ${HEADERS} ${SOURCES})
TARGET_LINK_LIBRARIES(coFFmpegEncoder ${EXTRA_LIBS} coUtil)
COVISE_INSTALL_TARGET(coFFmpegEncoder)
COVISE_INSTALL_HEADERS(coFFmpegEncoder ${HEADERS})