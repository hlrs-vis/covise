PROJECT(Plot CXX C)
IF(WIN32)
  RETURN()
ENDIF(WIN32)

COVISE_FIND_PACKAGE(X11)
COVISE_FIND_PACKAGE(Motif)

USING(MOTIF)

COVISE_FIND_PACKAGE(BISON)
IF(NOT BISON_FOUND)
   RETURN()
ENDIF()
#BISON_TARGET(pars pars.yacc ${CMAKE_CURRENT_BINARY_DIR}/pars.cpp COMPILE_FLAGS "--defines=${CMAKE_CURRENT_BINARY_DIR}/pars_yacc.h -y")
BISON_TARGET(pars pars.yacc ${CMAKE_CURRENT_BINARY_DIR}/pars.cpp COMPILE_FLAGS "-y")

SET(SOURCES
        PlotCommunication.cpp
        PlotError.cpp
        PlotPort.cpp
        as274c.cpp
        binio.cpp
        blockwin.cpp
        checkon.cpp
        chersh.cpp
        compute.cpp
        computils.cpp
        compwin.cpp
        comwin.cpp
        defaults.cpp
        draw.cpp
        drawticks.cpp
        drawwin.cpp
        eblockwin.cpp
        editpwin.cpp
        events.cpp
        files.cpp
        fileswin.cpp
        fit.cpp
        fourier.cpp
        framewin.cpp
        getparms.cpp
        graphutils.cpp
        graphutils2.cpp
        graphwin.cpp
        helpwin.cpp
        hotwin.cpp
        hp.cpp
        image_f_io.cpp
        io.cpp
        labelwin.cpp
        leaf.cpp
        lmdif1.cpp
        locatewin.cpp
        main.cpp
        malerts.cpp
        mif.cpp
        miscwin.cpp
        monwin.cpp
        motifutils.cpp
        nonlwin.cpp
        objutils.cpp
        pagewin.cpp
        params.cpp
        plotone.cpp
        printwin.cpp
        ps.cpp
        ptswin.cpp
        regionutils.cpp
        regionwin.cpp
        rotated.cpp
        setprops.cpp
        setutils.cpp
        setwin.cpp
        statuswin.cpp
        strwin.cpp
        stubs.cpp
        symwin.cpp
        tickwin.cpp
        utils.cpp
        worldwin.cpp
        xmgr.cpp
        xvlib.cpp
)

SET(EXTRASOURCES
   pars.yacc
)

ADD_DEFINITIONS("-DMOTIF -DHDEV=GR_PS_L -DMAXARR=20480 -DTEST")

IF(APPLE)
  SET(EXTRA_LIBS ${EXTRA_LIBS} /opt/X11/lib/libGL.dylib)
ELSE(APPLE)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${OPENGL_LIBRARIES})
ENDIF(APPLE)

ADD_COVISE_MODULE(Renderer Plot ${EXTRASOURCES} ${BISON_pars_OUTPUTS})
TARGET_LINK_LIBRARIES(Plot
  coAppl coCore coUtil
  ${EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT}
)

COVISE_WNOERROR(Plot)
COVISE_INSTALL_TARGET(Plot)
