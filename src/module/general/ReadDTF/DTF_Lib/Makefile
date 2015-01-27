
OBJECTFILES=bcrecords.o bcvc.o block.o file.o evaldata.o CellTypes.o \
DataTypes.o dataelement.o FaceTypes.o TopoTypes.o info.o libif.o \
libobject.o mesh.o misc.o patch.o polyzone.o polyzonedata.o sim.o \
simdata.o structzone.o surface.o unstrzone.o vcrecords.o virtualzone.o \
volume.o zone.o zonedata.o zoneinterface.o 

##########################################################
# All system-dependent UNIX compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:


# All common rules are in this central makefile.

include $(COVISEDIR)/src/make.rules

DTF_DIR=./cfdrc
DTF_INCLUDE=$(DTF_DIR)/include  


GREEN=\x1b\x5b32m
YELLOW=\x1b\x5b33m
DEFCOL=\x1b\x5b39m
BLUE=\x1b\x5b34m

PACKAGE=DTF_Lib

all: print $(OBJECTFILES)

print: 
	@echo -e "$(GREEN)COMPILING $(YELLOW)package $(PACKAGE) $(DEFCOL)"

%.o: %.cpp %.h
	@echo -e "$(GREEN)COMPILING $(YELLOW)$(PACKAGE)/$*.cpp $(BLUE)"
	@$(CC) $(CPPFLAGS) -I$(DTF_INCLUDE) -c $*.cpp
	@printf "$(DEFCOL)"

clean:
	@echo -e "$(GREEN)CLEANING $(YELLOW)$(PACKAGE) $(DEFCOL)"
	@rm -rf *.o *~ 
