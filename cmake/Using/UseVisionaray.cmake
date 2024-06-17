MACRO(USE_VISIONARAY)
    if(COVISE_USE_VISIONARAY)
        IF(NOT VISIONARAY_USED)
            set(VISIONARAY_INCLUDE_DIR "${COVISEDIR}/src/3rdparty/visionaray/include")

            if(COVISE_USE_CUDA)
                covise_find_package(CUDA)
            endif()

            if(NOT APPLE AND NOT WIN32)
                covise_find_package(PTHREAD REQUIRED)
            endif()

            include_directories(${VISIONARAY_INCLUDE_DIR})

            if (NOT APPLE AND NOT WIN32)
                include_directories(SYSTEM ${PTHREAD_INCLUDE_DIR})
                set(EXTRA_LIBS
                    ${EXTRA_LIBS}
                    ${PTHREAD_LIBRARY}
                    )
            endif()

            if(COVISE_USE_CUDA AND CUDA_FOUND)
                include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
                set(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_LIBRARIES})
            endif()

            SET(VISIONARAY_USED TRUE)
        ENDIF(NOT VISIONARAY_USED)
    else(COVISE_USE_VISIONARAY)
        if (${ARGC} LESS 1)
            using_message("Skipping because of disabled Visionaray")
            return()
        endif()
    endif(COVISE_USE_VISIONARAY)
ENDMACRO(USE_VISIONARAY)
