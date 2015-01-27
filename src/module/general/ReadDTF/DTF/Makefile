
OBJECTFILES=File.o Data.o Cell.o Node.o Sim.o Zone.o timesteps.o \
timestep.o grid.o ZD.o

##########################################################
# All system-dependent UNIX compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:


# All common rules are in this central makefile.

include $(COVISEDIR)/src/make.rules

DTF_DIR=../DTF_Lib/cfdrc
DTF_INCLUDE=$(DTF_DIR)/include  


GREEN=\x1b\x5b32m
YELLOW=\x1b\x5b33m
RED=\x1b\x5b31m
DEFCOL=\x1b\x5b39m
BLUE=\x1b\x5b34m

all: print $(OBJECTFILES)

print:
	@echo -e "$(GREEN)COMPILING $(YELLOW)package $(PACKAGE) $(DEFCOL)"

%.o: %.cpp %.h
	@echo -e "$(GREEN)COMPILING $(YELLOW)$(PACKAGE)/$*.cpp $(BLUE)"
	@$(CC) $(CPPFLAGS) -I$(DTF_INCLUDE) -c $*.cpp
	@printf "$(DEFCOL)"

timesteps.o: timesteps.cpp timesteps.h
	@echo -e "$(GREEN)COMPILING $(YELLOW)$(PACKAGE)/timesteps.cpp $(RED)"
	@$(CC) $(CPPFLAGS) -I$(DTF_INCLUDE) -c timesteps.cpp
	@printf "$(DEFCOL)"

timestep.o: timestep.cpp timestep.h
	@echo -e "$(GREEN)COMPILING $(YELLOW)$(PACKAGE)/timestep.cpp $(RED)"
	@$(CC) $(CPPFLAGS) -I$(DTF_INCLUDE) -c timestep.cpp
	@printf "$(DEFCOL)"

clean:
	@echo -e "$(GREEN)CLEANING $(YELLOW)$(PACKAGE) $(DEFCOL)"
	@rm -rf *.o *~ 
