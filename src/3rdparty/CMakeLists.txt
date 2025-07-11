set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${COVISEDIR}/${COVISE_ARCHSUFFIX}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${COVISEDIR}/${COVISE_ARCHSUFFIX}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${COVISEDIR}/${COVISE_ARCHSUFFIX}/lib")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${COVISEDIR}/${DBG_ARCHSUFFIX}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${COVISEDIR}/${DBG_ARCHSUFFIX}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${COVISEDIR}/${DBG_ARCHSUFFIX}/lib")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${COVISEDIR}/${DBG_ARCHSUFFIX}opt/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${COVISEDIR}/${DBG_ARCHSUFFIX}opt/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${COVISEDIR}/${DBG_ARCHSUFFIX}opt/lib")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${COVISEDIR}/${DBG_ARCHSUFFIX}opt/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${COVISEDIR}/${DBG_ARCHSUFFIX}opt/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${COVISEDIR}/${DBG_ARCHSUFFIX}opt/lib")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${COVISEDIR}/${DBG_ARCHSUFFIX}opt/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${COVISEDIR}/${DBG_ARCHSUFFIX}opt/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL "${COVISEDIR}/${DBG_ARCHSUFFIX}opt/lib")

SET(DESKVOX_USE_VISIONARAY ${COVISE_USE_VISIONARAY} CACHE BOOL "Visionaray for Virvo" FORCE)
SET(DESKVOX_USE_CUDA ${COVISE_USE_CUDA} CACHE BOOL "CUDA for Virvo" FORCE)
SET(DESKVOX_USE_QT4 OFF CACHE BOOL "Qt4 for Virvo" FORCE)
SET(DESKVOX_BUILD_VOX_DESK OFF CACHE BOOL "Qt GUI for Deskvox" FORCE)
SET(DESKVOX_USE_RAYREND_BSPLINE OFF)
SET(DESKVOX_USE_RAYREND_CSPLINE OFF)

if (COVISE_USE_VIRVO)
   using(GLEW optional)
   if (NOT GLEW_FOUND)
      message("Cannot build Virvo: no GLEW - maybe disable COVISE_USE_VIRVO?")
   endif()
   # search for Boost components required by Virvo with covise_find_package:
   # avoids the problem that Virvo does not find Boost in $EXTERNLIBS
   using(Boost optional)
   if (COVISE_USE_VISIONARAY)
      set(visionaray_DIR "${CMAKE_CURRENT_LIST_DIR}/visionaray/cmake")
   endif()
   if (NOT Boost_FOUND)
      message("Cannot build Virvo: no Boost - maybe disable COVISE_USE_VIRVO?")
   endif()
   set(saved_prefix_path ${CMAKE_PREFIX_PATH})
   set(CMAKE_PREFIX_PATH "$ENV{EXTERNLIBS}/volpack" ${CMAKE_PREFIX_PATH})
   set(CMAKE_PREFIX_PATH "$ENV{EXTERNLIBS}/gdcm" ${CMAKE_PREFIX_PATH})
   set(CMAKE_PREFIX_PATH "$ENV{EXTERNLIBS}/nifti" ${CMAKE_PREFIX_PATH})
   set(CMAKE_PREFIX_PATH "$ENV{EXTERNLIBS}/teem" ${CMAKE_PREFIX_PATH})
   add_definitions(-DHAVE_CONFIG_H)
   ADD_SUBDIRECTORY(deskvox)
   set(${CMAKE_PREFIX_PATH} ${saved_prefix_path})
   covise_export_target(virvo_fileio)
   covise_export_target(virvo)
   if (OPENSCENEGRAPH_FOUND)
      covise_export_target(virvo_osg)
   else()
      message("Cannot build virvo_osg: no OpenSceneGraph - maybe disable COVISE_USE_VIRVO?")
   endif()
endif()

add_library(exprtk INTERFACE)
target_include_directories(exprtk INTERFACE ${CMAKE_CURRENT_LISTS_DIR}/exprtk)
target_include_directories(exprtk INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/exprtk>
    $<INSTALL_INTERFACE:include>
)
if(MSVC)
  target_compile_options(exprtk INTERFACE /bigobj)
endif()

# Export the exprtk target
install(TARGETS exprtk EXPORT exprtkTargets)

# Install exprtk header file
COVISE_INSTALL_HEADERS(${CMAKE_CURRENT_LISTS_DIR}/exprtk/exprtk.hpp)

# Export the exprtkTargets export set
install(EXPORT exprtkTargets
    FILE exprtkTargets.cmake
    NAMESPACE exprtk::
    DESTINATION lib/cmake/exprtk
)

covise_export_target(exprtk)

if(NOT TARGET nlohmann_json::nlohmann_json)
    ADD_SUBDIRECTORY(nlohmann_json)
    install(TARGETS nlohmann_json EXPORT covise-targets)
    COVISE_EXPORT_TARGET(nlohmann_json)
endif()

set(CROW_USE_BOOST ON CACHE BOOL "Use Boost.Asio for Crow" FORCE)
add_subdirectory(crow)
