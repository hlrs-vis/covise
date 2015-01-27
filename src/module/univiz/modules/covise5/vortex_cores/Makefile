##########################################################
## Description: Makefile Template for vortex core lines ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Andreas Werner              Date: 10.01.2000 ##
##########################################################

                          # name of the module executable
APPLICATION = VortexCores
                          # name of the module category
CATEGORY    = Univiz
                          # needed for creating distribution tar-files
PROJECT	    = Univiz

# put all object files that are needed to build the module here,
# e.g.   $(OBJDIR)/object.o

OBJECTS         = $(OBJDIR)/unstructured.o $(OBJDIR)/unigeom.o $(OBJDIR)/unisys.o $(OBJDIR)/computeVortexCores.o 

# $(APPLICATION) is also the name of the main source and include file!!!)

##########################################################

# All system-dependent UNIX compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:

# added

UNIVIZ_LIBS    = ../../../libs
UNIVIZ_MOD_IMPL = ../../impl

CPPFLAGS += -DCOVISE -DCOVISE5
CPPFLAGS += -I$(UNIVIZ_LIBS)/linalg -I$(UNIVIZ_LIBS)/unstructured -I$(UNIVIZ_LIBS)/unigeom -I$(UNIVIZ_LIBS)/unisys -I$(UNIVIZ_MOD_IMPL)/vortex_cores

# All common rules are in this central makefile.

include $(COVISEDIR)/src/make.rules

############### link rule for application module #########
# can be modified (e.g. for dependencies or libraries)
##########################################################

$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION): $(OBJDIR)/$(APPLICATION).o $(OBJECTS)
	  $(CC) $(CPPFLAGS) -o $@ $(OBJECTS) $(OBJDIR)/$(APPLICATION).o $(IDENTOBJ) \
	        $(COVISE_LIBDIR) $(COVISEAPI) $(APPLLIB) $(COVISELIB)

############### general object rule ######################
# can be modified (e.g. for dependencies)
##########################################################

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp $(APPLICATION).h
	$(CC) $(CPPFLAGS) -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	$(POSTCC)

############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################

#$(OBJDIR)/object.o: object.cpp object.h $(APPLICATION).h
#	$(CC) $(CPPFLAGS) -c object.cpp -o $(OBJDIR)/object.o
#	$(POSTCC)


# added

$(OBJDIR)/unstructured.o: $(UNIVIZ_LIBS)/unstructured/unstructured.cpp $(UNIVIZ_LIBS)/unstructured/unstructured.h
	$(CC) $(CPPFLAGS) -c $(UNIVIZ_LIBS)/unstructured/unstructured.cpp -o $(OBJDIR)/unstructured.o
	$(POSTCC)

$(OBJDIR)/unigeom.o: $(UNIVIZ_LIBS)/unigeom/unigeom.cpp $(UNIVIZ_LIBS)/unigeom/unigeom.h
	$(CC) $(CPPFLAGS) -c $(UNIVIZ_LIBS)/unigeom/unigeom.cpp -o $(OBJDIR)/unigeom.o
	$(POSTCC)

$(OBJDIR)/unisys.o: $(UNIVIZ_LIBS)/unisys/unisys.cpp $(UNIVIZ_LIBS)/unisys/unisys.h
	$(CC) $(CPPFLAGS) -c $(UNIVIZ_LIBS)/unisys/unisys.cpp -o $(OBJDIR)/unisys.o
	$(POSTCC)

$(OBJDIR)/computeVortexCores.o: $(UNIVIZ_MOD_IMPL)/vortex_cores/computeVortexCores.cpp $(UNIVIZ_MOD_IMPL)/vortex_cores/computeVortexCores.h
	$(CC) $(CPPFLAGS) -c $(UNIVIZ_MOD_IMPL)/vortex_cores/computeVortexCores.cpp -o $(OBJDIR)/computeVortexCores.o
	$(POSTCC)

test:
	covise -e test.net
