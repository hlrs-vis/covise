


SET(HEADERS
arrays.h
classext.h
coviseInterface.h
error.h
fortran.h
grid.h
laser.h
main.h
material.h
matvect.h
raytrace.h
simul.h
solid.h
solve.h
Trans3D.h
trans3DInterface.h

)
SET(SOURCES
  classext.cpp
coviseInterface.cpp
error.cpp
fortran.cpp
grid.cpp
laser.cpp
main.cpp
material.cpp
matvect.cpp
raytrace.cpp
simul.cpp
solid.cpp
solve.cpp
Trans3D.cpp
trans3DInterface.cpp
)
add_definitions("-DCOVISE")
covise_add_module(Simulation Trans3d ${EXTRASOURCES} ${SOURCES} ${HEADERS})