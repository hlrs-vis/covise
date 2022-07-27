using(FFMPEG)



if (NOT FFMPEG_FOUND)
    return()
endif()

SET(HEADERS stream.h)
SET(SOURCES stream.cpp)

  
ADD_DEFINITIONS("-DHAVE_CODEC_CTX_PTR")
  


cover_add_plugin(Stream)
target_link_libraries(Stream coFFMPEG_Util)


