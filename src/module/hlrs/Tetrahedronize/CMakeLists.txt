SET(HEADERS
)
SET(SOURCES
  Grid.cpp
  Tetrahedronize.cpp
)
ADD_COVISE_MODULE(Obsolete Tetrahedronize ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(Tetrahedronize coApi coAppl coCore)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
target_compile_options(Tetrahedronize PRIVATE -Wno-error=alloc-size-larger-than=)
endif()
COVISE_INSTALL_TARGET(Tetrahedronize)
