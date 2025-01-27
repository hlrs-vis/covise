SET(HEADERS
   DTrackDataTypes.hpp
   DTrackNet.hpp
   DTrackSDK.hpp
   DTrackParse.hpp
   DTrackParser.hpp
   DTrackDriver.h
)

SET(SOURCES
   DTrackNet.cpp
   DTrackParse.cpp
   DTrackParser.cpp
   DTrackSDK.cpp
   DTrackDriver.cpp
)

vive_add_plugin(vvInput_dtrack)
