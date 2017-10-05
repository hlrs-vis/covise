USING(BOOST)
SET(HEADERS
  ReadFOAM.h
)
SET(SOURCES
  ReadFOAM.cpp
  foamtoolbox.cpp
)
ADD_COVISE_MODULE(IO ReadFoam ${EXTRASOURCES})
if(NOT MSVC)
    add_covise_compile_flags(ReadFoam "-Wno-error=deprecated-declarations")
endif()
TARGET_LINK_LIBRARIES(ReadFoam coApi coAppl coCore ${EXTRA_LIBS})
COVISE_INSTALL_TARGET(ReadFoam)
