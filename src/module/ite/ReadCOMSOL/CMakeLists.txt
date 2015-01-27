IF(WIN32)
SET(HEADERS
  projects/covise/ReadCOMSOL/sourcecode/readcomsol.h
  projects/interfaces/comsol/sourcecode/communicationcomsolstandard.h
  projects/interfaces/comsol/sourcecode/comsolphysicsstandard.h
)
SET(SOURCES
projects/covise/ReadCOMSOL/sourcecode/readcomsol.cpp
projects/covise/ReadCOMSOL/sourcecode/readcomsolcomsol.cpp
projects/covise/ReadCOMSOL/sourcecode/readcomsoldialog.cpp
projects/covise/ReadCOMSOL/sourcecode/readcomsolinit.cpp
projects/covise/ReadCOMSOL/sourcecode/readcomsoloutputports.cpp
projects/covise/ReadCOMSOL/sourcecode/readcomsoltools.cpp
projects/interfaces/comsol/sourcecode/communicationcomsol.cxx
projects/interfaces/comsol/sourcecode/communicationcomsolstandard.cpp
projects/interfaces/comsol/sourcecode/comsolphysics.cxx
projects/interfaces/comsol/sourcecode/comsolphysicsstandard.cpp
projects/interfaces/matlab/sourcecode/interfacematlab.cxx
projects/interfaces/matlab/sourcecode/interfacematlabcom.cxx
projects/interfaces/matlab/sourcecode/interfacematlabengine.cxx
projects/interfaces/matlab/sourcecode/interfacematlabcom.cxx
projects/interfaces/matlab/sourcecode/interfacematlabengine.cxx
)
ADD_COVISE_MODULE(ITE ReadCOMSOL ${EXTRASOURCES})
TARGET_LINK_LIBRARIES(ReadCOMSOL coApi coAppl coCore)
COVISE_INSTALL_TARGET(ReadCOMSOL)
ENDIF(WIN32)
