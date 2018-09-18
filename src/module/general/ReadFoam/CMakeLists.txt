if (APPLE)
   # find keg-only libarchive from homebrew
   set(LibArchive_ROOT /usr/local/opt/libarchive)
endif()
covise_find_package(LibArchive)
if (LibArchive_FOUND)
   include_directories(SYSTEM ${LibArchive_INCLUDE_DIRS})
   add_definitions(-DHAVE_LIBARCHIVE)
endif()

using(BOOST)

set(HEADERS
  ReadFOAM.h
)

set(SOURCES
  ReadFOAM.cpp
  foamtoolbox.cpp
  archivemodel.cpp
)

add_covise_module(IO ReadFoam ${EXTRASOURCES})
if(NOT MSVC)
    add_covise_compile_flags(ReadFoam "-Wno-error=deprecated-declarations")
endif()
target_link_libraries(ReadFoam coApi coAppl coCore ${EXTRA_LIBS})
if (LibArchive_FOUND)
   target_link_libraries(ReadFoam ${LibArchive_LIBRARIES})
endif()

covise_install_target(ReadFoam)
