#
# Makefile for dataDict
#
#G=-g -Wall -Wno-deprecated
G=-O3 -Wall -Wno-deprecated

INC = -I../linalg
CFLAGS = $(INC) $(AOPTCFLAGS) $(LOCAL_CFLAGS) $(G)

dataDict.o: dataDict.cpp dataDict.h
	$(CXX) $(CFLAGS) -c dataDict.cpp

clean:
	rm *.o
