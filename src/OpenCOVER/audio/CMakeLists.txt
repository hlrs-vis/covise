covise_find_package(glm)
covise_find_package(OpenAL)
covise_find_package(ALUT)

# Workaround for GLM 0.9.9 broken glmTargets.cmake
if(NOT TARGET glm::glm AND GLM_INCLUDE_DIRS)
    add_library(glm::glm INTERFACE IMPORTED)
    set_target_properties(glm::glm PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIRS}"
    )
endif()

# glm: allow .xyz() and similar access
add_definitions(-DGLM_FORCE_SWIZZLE -DGLM_ENABLE_EXPERIMENTAL)

set(AUDIO_HEADERS 
    AlutContext.h
    Audio.h
    Listener.h
    Player.h
)

set(AUDIO_SOURCES 
    AlutContext.cpp
    Audio.cpp
    Listener.cpp
    Player.cpp
)

if(ALUT_FOUND AND (TARGET OpenAL::OpenAL))
    set(HAVE_AUDIO TRUE)

    list(APPEND AUDIO_HEADERS PlayerOpenAL.h PlayerAServer.h)
    list(APPEND AUDIO_SOURCES PlayerOpenAL.cpp PlayerAServer.cpp)
else()
    set(HAVE_AUDIO FALSE)
endif()

add_covise_library(coVRAudio SHARED ${AUDIO_SOURCES} ${AUDIO_HEADERS})

target_link_libraries(coVRAudio
    coOpenConfig
    coConfig
    coNet
    ${COVISE_UTIL_LIBRARY}
    glm::glm
)

if (HAVE_AUDIO)
    target_link_libraries(coVRAudio ${ALUT_LIBRARIES} OpenAL::OpenAL)
    target_include_directories(coVRAudio PRIVATE ${ALUT_INCLUDE_DIR} ${OPENAL_INCLUDE_DIR})
    target_compile_definitions(coVRAudio PRIVATE -DHAVE_AUDIO)
endif()

# silence macos deprecation warnings
if(APPLE)
    target_compile_definitions(coVRAudio PRIVATE -DOPENAL_DEPRECATED=)
endif()

# TODO: document why MSCV needs to link to bcrypt
if(MSVC)
    target_link_libraries(coVRAudio bcrypt.lib)
endif()

covise_install_target(coVRAudio)
covise_install_headers(audio ${AUDIO_HEADERS})
