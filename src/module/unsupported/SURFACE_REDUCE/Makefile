##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Uwe Woessner, Andreas Wierse  Date: 29.08.97 ##
##########################################################

                          # name of the module executable
APPLICATION 	= SurfaceReduce
                          # name of the module category
CATEGORY	= Filter
                          # needed for creating distribution tar-files
PROJECT		= General

# put all object files that are needed to build the module here:

OBJECTS         = $(OBJDIR)/SurfEdge.o \
                  $(OBJDIR)/HandleSet.o \
		  $(OBJDIR)/Surface.o 

#ADDFLAG =
ADDFLAG = -g
 
# $(APPLICATION) is also the name of the main source and include file!!!)

##########################################################

# modify this for your own path if necessary
# COVISEDIR = ../../../..

# All common UNIX compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:


# All common rules are in this central makefile.

include $(COVISEDIR)/src/make.rules

############### link rule for application module #########
# can be modified (e.g. for dependencies or libraries)
##########################################################

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJDIR)/$(APPLICATION).o $(OBJECTS)
	  $(CC) $(CPPFLAGS) $(ADDFLAG) -o $@ $(OBJECTS) $(OBJDIR)/$(APPLICATION).o \
	  $(APPLLIB) $(LDLIBS) -lm

############### general object rule ######################
# can be modified (e.g. for dependencies)
##########################################################

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp $(APPLICATION).h
	$(CC) $(CPPFLAGS) $(ADDFLAG) -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	$(POSTCC)

############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################

$(OBJDIR)/HandleSet.o: HandleSet.cpp $(APPLICATION).h
	$(CC) $(CPPFLAGS) $(ADDFLAG) -c HandleSet.cpp -o $(OBJDIR)/HandleSet.o
	$(POSTCC)

$(OBJDIR)/PQ.o: PQ.cpp $(APPLICATION).h
	$(CC) $(CPPFLAGS) $(ADDFLAG) -c PQ.cpp -o $(OBJDIR)/PQ.o
	$(POSTCC)

$(OBJDIR)/Surface.o: Surface.cpp $(APPLICATION).h
#	$(CC) $(CPPFLAGS) $(ADDFLAG) +d -c Surface.cpp -o $(OBJDIR)/Surface.o
	$(CC) $(CPPFLAGS) $(ADDFLAG) -c Surface.cpp -o $(OBJDIR)/Surface.o
	$(POSTCC)
	
$(OBJDIR)/SurfEdge.o: SurfEdge.cpp $(APPLICATION).h
#	$(CC) $(CPPFLAGS) $(ADDFLAG) +d -c SurfEdge.cpp -o $(OBJDIR)/SurfEdge.o
	$(CC) $(CPPFLAGS) $(ADDFLAG) -c SurfEdge.cpp -o $(OBJDIR)/SurfEdge.o
	$(POSTCC)
