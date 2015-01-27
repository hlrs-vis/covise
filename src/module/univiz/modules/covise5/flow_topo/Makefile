##########################################################
## Description: Makefile Template for HELLO program     ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Andreas Werner              Date: 10.01.2000 ##
##########################################################

                          # name of the module executable
APPLICATION = FlowTopo
                          # name of the module category
CATEGORY	   = Univiz
                          # needed for creating distribution tar-files
PROJECT		= Univiz

# put all object files that are needed to build the module here,
# e.g.   $(OBJDIR)/object.o

OBJECTS         = $(OBJDIR)/unstructured.o $(OBJDIR)/unifield.o $(OBJDIR)/unisys.o

# $(APPLICATION) is also the name of the main source and include file!!!)

##########################################################

# All system-dependent UNIX compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:

UNIVIZ_LIBS    = ../../../libs
UNIVIZ_MOD_IMPL = ../../impl

CPPFLAGS += -DCOVISE -DCOVISE5
CPPFLAGS += -I$(UNIVIZ_LIBS)/linalg -I$(UNIVIZ_LIBS)/unstructured -I$(UNIVIZ_LIBS)/unifield -I$(UNIVIZ_LIBS)/unisys -I$(UNIVIZ_MOD_IMPL)/flow_topo

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

# by sadlo
$(OBJDIR)/unstructured.o: $(UNIVIZ_LIBS)/unstructured/unstructured.cpp $(UNIVIZ_LIBS)/unstructured/unstructured.h
	$(CC) $(CPPFLAGS) -c $(UNIVIZ_LIBS)/unstructured/unstructured.cpp -o $(OBJDIR)/unstructured.o
	$(POSTCC)

$(OBJDIR)/unifield.o: $(UNIVIZ_LIBS)/unifield/unifield.cpp $(UNIVIZ_LIBS)/unifield/unifield.h
	$(CC) $(CPPFLAGS) -c $(UNIVIZ_LIBS)/unifield/unifield.cpp -o $(OBJDIR)/unifield.o
	$(POSTCC)

$(OBJDIR)/unisys.o: $(UNIVIZ_LIBS)/unisys/unisys.cpp $(UNIVIZ_LIBS)/unisys/unisys.h
	$(CC) $(CPPFLAGS) -c $(UNIVIZ_LIBS)/unisys/unisys.cpp -o $(OBJDIR)/unisys.o
	$(POSTCC)

test:
	covise -e test.net
