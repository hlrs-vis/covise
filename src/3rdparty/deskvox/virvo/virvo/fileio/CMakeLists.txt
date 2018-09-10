find_package(Boost COMPONENTS filesystem serialization system REQUIRED)
find_package(Nifti)
find_package(Teem)

if(DESKVOX_USE_GDCM)
    find_package(GDCM)
    if (GDCM_FOUND)
        include(${GDCM_USE_FILE})
    endif()
endif()

deskvox_use_package(Boost)
if(DESKVOX_USE_CUDA)
find_package(CUDA)
deskvox_use_package(CUDA)
endif()
if(DESKVOX_USE_GDCM)
deskvox_use_package(GDCM)
endif()
deskvox_use_package(Nifti)
deskvox_use_package(Teem)

set(VIRVO_FILEIO_HEADERS
    ${VIRVO_SOURCE_DIR}/private/vvlog.h
    ${VIRVO_SOURCE_DIR}/vvclock.h
    ${VIRVO_SOURCE_DIR}/vvcolor.h
    ${VIRVO_SOURCE_DIR}/vvdebugmsg.h
    ${VIRVO_SOURCE_DIR}/vvdicom.h
    ${VIRVO_SOURCE_DIR}/vvfileio.h
    ${VIRVO_SOURCE_DIR}/vvtokenizer.h
    ${VIRVO_SOURCE_DIR}/vvtfwidget.h
    ${VIRVO_SOURCE_DIR}/vvtoolshed.h
    ${VIRVO_SOURCE_DIR}/vvtransfunc.h
    ${VIRVO_SOURCE_DIR}/vvvoldesc.h

    gdcm.h
    feature.h
    nifti.h
    nrrd.h
)


set(VIRVO_FILEIO_SOURCES
    ${VIRVO_SOURCE_DIR}/private/vvlog.cpp
    ${VIRVO_SOURCE_DIR}/vvclock.cpp
    ${VIRVO_SOURCE_DIR}/vvcolor.cpp
    ${VIRVO_SOURCE_DIR}/vvdebugmsg.cpp
    ${VIRVO_SOURCE_DIR}/vvdicom.cpp
    ${VIRVO_SOURCE_DIR}/vvfileio.cpp
    ${VIRVO_SOURCE_DIR}/vvtokenizer.cpp
    ${VIRVO_SOURCE_DIR}/vvtfwidget.cpp
    ${VIRVO_SOURCE_DIR}/vvtoolshed.cpp
    ${VIRVO_SOURCE_DIR}/vvtransfunc.cpp
    ${VIRVO_SOURCE_DIR}/vvvoldesc.cpp

    gdcm.cpp
    feature.cpp
    nifti.cpp
    nrrd.cpp
)

if(DESKVOX_USE_CUDA)
    set(VIRVO_FILEIO_SOURCES ${VIRVO_FILEIO_SOURCES}
        ${VIRVO_SOURCE_DIR}/vvcudatransfunc.cpp
    )
deskvox_cuda_compiles(VIRVO_FILEIO_CUDA_SOURCES
    ${VIRVO_SOURCE_DIR}/vvcudatransfunc.cu
)
endif()

set(VIRVO_FILEIO_DEFINITIONS
    HAVE_CONFIG_H
)
if(DESKVOX_USE_GDCM AND GDCM_FOUND)
   deskvox_link_libraries(gdcmDSED gdcmMSFF gdcmDICT gdcmCommon)
endif()

deskvox_add_library(virvo_fileio
    ${VIRVO_FILEIO_HEADERS}
    ${VIRVO_FILEIO_SOURCES}
    ${VIRVO_FILEIO_CUDA_SOURCES}
)

set_target_properties(virvo_fileio
    PROPERTIES
        COMPILE_DEFINITIONS ${VIRVO_FILEIO_DEFINITIONS}
)
