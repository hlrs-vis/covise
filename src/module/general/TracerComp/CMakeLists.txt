USING(Pthreads)

ADD_DEFINITIONS(-D_COMPLEX_MODULE_)

INCLUDE_DIRECTORIES(
  ../Tracer
)

SET(HEADERS
  
)

SET(SOURCES
  ../Tracer/BBoxAdmin.cpp
  ../Tracer/HTask.cpp
  ../Tracer/PPathline.cpp
  ../Tracer/PPathlineStat.cpp
  ../Tracer/PPathlineStatInterpolation.cpp
  ../Tracer/PPathlineStatNoControl.cpp
  ../Tracer/PStreamline.cpp
  ../Tracer/PTask.cpp
  ../Tracer/PTraceline.cpp
  ../Tracer/Pathlines.cpp
  ../Tracer/PathlinesStat.cpp
  ../Tracer/Streaklines.cpp
  ../Tracer/Streamlines.cpp
  ../Tracer/Tracelines.cpp
  ../Tracer/Tracer.cpp
  ../Tracer/Tracer1.cpp
  ../Tracer/PointsParser.cpp
)

SET(EXTRASOURCES
  ../Tracer/BBoxAdmin.h
  ../Tracer/Fifo.h
  ../Tracer/HTask.h
  ../Tracer/PPathline.h
  ../Tracer/PPathlineStat.h
  ../Tracer/PPathlineStatInterpolation.h
  ../Tracer/PPathlineStatNoControl.h
  ../Tracer/PStreamline.h
  ../Tracer/PTask.h
  ../Tracer/PTraceline.h
  ../Tracer/Pathlines.h
  ../Tracer/PathlinesStat.h
  ../Tracer/PointsLexer.h
  ../Tracer/Streaklines.h
  ../Tracer/Streamlines.h
  ../Tracer/Tracelines.h
  ../Tracer/Tracer.h
)

ADD_COVISE_MODULE(Mapper TracerComp ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(TracerComp  coAlg coApi coAppl coCore coUtil ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(TracerComp)
