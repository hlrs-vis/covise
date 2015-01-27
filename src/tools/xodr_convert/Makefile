XERCESCINCLUDE = -I$(EXTERNLIBS)/xerces/include -I/usr/local/include
XERCESCLIB = -L$(EXTERNLIBS)/xerces/lib -lxerces-c -L/usr/local/lib
OSGINCLUDE = -I$(EXTERNLIBS)/OpenSceneGraph/include
OSGLIB = -L$(EXTERNLIBS)/OpenSceneGraph/lib -losg -losgViewer

LIBS = $(XERCESCLIB) $(OSGLIB)
INCLUDE = $(OSGINCLUDE)

CPP = c++ -g $(INCLUDE)
DEFINES = -DHAVE_OSG

ALLOBJECTS = RoadSystem.o Element.o Types.o Tarmac.o Road.o LaneSection.o Lane.o Junction.o 

all: RoadSystem Element Types Tarmac Road LaneSection Lane Junction



RoadSystem: RoadSystem.h RoadSystem.cpp
	$(CPP) $(DEFINES) -c $(XERCESCINCLUDE) -o RoadSystem.o RoadSystem.cpp

Element: Element.h Element.cpp
	$(CPP) $(DEFINES) -c -o Element.o Element.cpp

Types: Types.h Types.cpp
	$(CPP) $(DEFINES) -c -o Types.o Types.cpp

Tarmac: Tarmac.h Tarmac.cpp Element
	$(CPP) $(DEFINES) -c -o Tarmac.o Tarmac.cpp

Road: Road.h Road.cpp Tarmac
	$(CPP) $(DEFINES) -c -o Road.o Road.cpp

LaneSection: LaneSection.h LaneSection.cpp Lane
	$(CPP) $(DEFINES) -c -o LaneSection.o LaneSection.cpp

Lane: Lane.h Lane.cpp Types
	$(CPP) $(DEFINES) -c -o Lane.o Lane.cpp

Junction: Junction.h Junction.cpp Road Tarmac
	$(CPP) $(DEFINES) -c -o Junction.o Junction.cpp


xodr_convert: xodr_convert.cpp all
	$(CPP) $(DEFINES) -o xodr_convert $(ALLOBJECTS) $(LIBS) xodr_convert.cpp 


