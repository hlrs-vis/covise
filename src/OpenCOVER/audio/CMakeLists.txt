FIND_PACKAGE(glm REQUIRED)
FIND_PACKAGE(OpenAL REQUIRED)
FIND_PACKAGE(ALUT REQUIRED)

set(AUDIO_SOURCES 
    AlutContext.h
    Audio.h
    Listener.h
    Player.h
    PlayerAServer.h
    PlayerOpenAL.h
    PlayerOsc.h
)

SET(AUDIO_SOURCES 
    AlutContext.cpp
    Audio.cpp
    Listener.cpp
    Player.cpp
    PlayerAServer.cpp
    PlayerOpenAL.cpp
    PlayerOsc.cpp
)

ADD_COVISE_LIBRARY(coVRAudio SHARED ${AUDIO_SOURCES} ${AUDIO_HEADERS})

# glm: allow .xyz() and similar access
ADD_DEFINITIONS(-DGLM_FORCE_SWIZZLE -DGLM_ENABLE_EXPERIMENTAL)

TARGET_LINK_LIBRARIES(coVRAudio
    coOpenConfig
    coConfig
    coNet
    ${COVISE_UTIL_LIBRARY}
    ${OPENAL_LIBRARY}
    ${ALUT_LIBRARIES}
    glm::glm
)
target_include_directories(coVRAudio 
    PRIVATE 
    ${OPENAL_INCLUDE_DIR}
    ${ALUT_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/oscpp/include"
)

COVISE_INSTALL_TARGET(coVRAudio)
COVISE_INSTALL_HEADERS(audio ${AUDIO_HEADERS})
