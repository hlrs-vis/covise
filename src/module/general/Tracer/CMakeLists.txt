USING(Pthreads)

SET(HEADERS
  
)

SET(SOURCES
  BBoxAdmin.cpp
  HTask.cpp
  PPathline.cpp
  PPathlineStat.cpp
  PPathlineStatInterpolation.cpp
  PPathlineStatNoControl.cpp
  PStreamline.cpp
  PTask.cpp
  PTraceline.cpp
  Pathlines.cpp
  PathlinesStat.cpp
  Streaklines.cpp
  Streamlines.cpp
  Tracelines.cpp
  Tracer.cpp
  Tracer1.cpp
)

SET(EXTRASOURCES
  BBoxAdmin.h
  Fifo.h
  HTask.h
  PPathline.h
  PPathlineStat.h
  PPathlineStatInterpolation.h
  PPathlineStatNoControl.h
  PStreamline.h
  PTask.h
  PTraceline.h
  Pathlines.h
  PathlinesStat.h
  PointsLexer.h
  Streaklines.h
  Streamlines.h
  Tracelines.h
  Tracer.h
)

ADD_COVISE_MODULE(Mapper Tracer ${EXTRASOURCES} )
TARGET_LINK_LIBRARIES(Tracer  coApi coAppl coCore coUtil ${EXTRA_LIBS})

COVISE_INSTALL_TARGET(Tracer)
