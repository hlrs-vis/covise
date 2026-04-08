FIND_PACKAGE(glm REQUIRED)
FIND_PACKAGE(ALUT REQUIRED)
USING(OpenSceneGraph)
USING(OpenAL)

# Workaround for GLM 0.9.9 broken glmTargets.cmake
if(NOT TARGET glm::glm)
    add_library(glm::glm INTERFACE IMPORTED)
    set_target_properties(glm::glm PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIRS}"
    )
endif()

set(AUDIO_HEADERS 
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
if(MSVC)
set(BCRYPT bcrypt.lib)
else()
set(BCRYPT)
endif()

TARGET_LINK_LIBRARIES(coVRAudio
    coOpenConfig
    coConfig
    coNet
    ${COVISE_UTIL_LIBRARY}
    ${ALUT_LIBRARIES}
    OpenAL::OpenAL
    glm::glm
	${BCRYPT}
)
target_include_directories(coVRAudio 
    PRIVATE 
    ${ALUT_INCLUDE_DIR}
    ${OPENAL_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/oscpp/include"
)
if(APPLE)
    # silence macos deprecation warnings
    target_compile_definitions(coVRAudio PRIVATE -DOPENAL_DEPRECATED=)
endif()

COVISE_INSTALL_TARGET(coVRAudio)
COVISE_INSTALL_HEADERS(audio ${AUDIO_HEADERS})
