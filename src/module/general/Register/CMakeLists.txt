
USING(ITK)

SET(HEADERS
  
)

SET(SOURCES
  Register.cpp
)

SET(EXTRASOURCES
  Register.h
)

ADD_COVISE_MODULE(Filter Register ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Register coApi coAppl coCore ${ITK_LIBRARIES})

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_covise_compile_flags(Register "-Wno-error=inconsistent-missing-override")
endif()


COVISE_INSTALL_TARGET(Register)
