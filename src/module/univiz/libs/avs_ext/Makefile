#
# Makefile for avs_ext
#
#G=-g -Wall -Wno-deprecated
G=-O3 -Wall -Wno-deprecated
include ../../sys/avs/defs.$(AVSMACHINE).mk

INC_FILE=$(AVS_PATH)/include/Makeinclude
INC = -I$(AVS_PATH)/include -I../linalg
CFLAGS = $(INC) $(AOPTCFLAGS) $(LOCAL_CFLAGS) $(G)

avs_ext.o: avs_ext.cpp avs_ext.h
	$(CXX) $(CFLAGS) -c avs_ext.cpp

clean:
	rm *.o
