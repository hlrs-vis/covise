IF(NOT OPENSCENEGRAPH_FOUND)
  RETURN()
ENDIF()
USING(VTK optional)

ADD_DEFINITIONS(

)

INCLUDE_DIRECTORIES(
)

SET(HEADERS
  
)

SET(SOURCES
  OpenCOVERMain.cpp
)

SET(EXTRASOURCES
  
)

IF(APPLE)
   SET(EXTRA_LIBS ${EXTRA_LIBS} "-framework AppKit")
   set(SOURCES ${SOURCES} osx-util.mm)
ENDIF(APPLE)

add_covise_module(Renderer OpenCOVER )
# old CONFIG: qt warn_on release incremental link_prl yacrenderer wnooverloadedvirtual qt osg tiff colib coutil coappl covrbclient werror opencoverlib osgvrui thread ssd wiimote cogrmsg jpeg
# old LIBS: 
# old links: 

TARGET_LINK_LIBRARIES(OpenCOVER  coOpenCOVER ${COVISE_VRBCLIENT_LIBRARY}
   ${COVISE_GRMSG_LIBRARY} ${COVISE_UTIL_LIBRARY} ${EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT} )
qt_use_modules(OpenCOVER Core)



COVISE_INSTALL_TARGET(OpenCOVER)
IF (MSVC)
MESSAGE("Adding Manifes to OpenCOVER")
ADD_CUSTOM_COMMAND(
TARGET OpenCOVER
POST_BUILD
COMMAND "mt.exe" -manifest "${CMAKE_CURRENT_SOURCE_DIR}[url=file://\\dpiaware.manifest]\\dpiaware.manifest[/url]" -inputresource:"$<TARGET_FILE:OpenCOVER>";#1 -outputresource:"$<TARGET_FILE:OpenCOVER>";#1
COMMENT "Adding DPI-aware manifest..."
)
ENDIF(MSVC) 
IF(UNIX)
  ADD_CUSTOM_COMMAND(TARGET OpenCOVER 
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${COVISE_DESTDIR}/${COVISE_ARCHSUFFIX}/bin/opencover
    COMMAND ${CMAKE_COMMAND} -E create_symlink Renderer/OpenCOVER ${COVISE_DESTDIR}/${COVISE_ARCHSUFFIX}/bin/opencover)

  IF(COVISE_PLACE_BINARIES_INSOURCE)
    ADD_CUSTOM_COMMAND(TARGET OpenCOVER 
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E remove -f $<TARGET_FILE_DIR:OpenCOVER>/../opencover
      COMMAND ${CMAKE_COMMAND} -E create_symlink Renderer/OpenCOVER $<TARGET_FILE_DIR:OpenCOVER>/../opencover)
  ENDIF(COVISE_PLACE_BINARIES_INSOURCE)

  INSTALL(FILES ${COVISE_DESTDIR}/${COVISE_ARCHSUFFIX}/bin/opencover DESTINATION ${COVISE_ARCHSUFFIX}/bin)
ENDIF()


