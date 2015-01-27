INCLUDES   	= -I.

COMPILER_FLAGS  = -Wno-deprecated -g -fopenmp
COMPILER        = g++

OBJ_FILES       = PointSort.o PointConvert.o

EXEC            = PointConvert

all:		PointSort PointConvert

PointSort:        PointSort.o
		$(COMPILER) $(COMPILER_FLAGS) $(INCLUDES) -o PointSort PointSort.o

PointConvert:        PointConvert.o
		$(COMPILER) $(COMPILER_FLAGS) $(INCLUDES) -o PointConvert PointConvert.o

PointSort.o: 	PointSort.cpp
		$(COMPILER) -c $(COMPILER_FLAGS) $(INCLUDES) PointSort.cpp

PointConvert.o: 	PointConvert.cpp
		$(COMPILER) -c $(COMPILER_FLAGS) $(INCLUDES) PointConvert.cpp
