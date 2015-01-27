
OBJECT_FILES=OutputManagerBase.o OutputManager.o \
EnumTypes.o classinfo.o classmanager.o baseobject.o \
statisticmanager.o statisticmanagerbase.o Helper.o

PACKAGE=Tools

##########################################################
# All system-dependent UNIX compiler flags and options are in
# this central makefile.

include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)

# if you want to modify settings of variables that are set in the
# central makefile, do this here:


# All common rules are in this central makefile.

include $(COVISEDIR)/src/make.rules

# define some colors for funny outputs until covise starts to gag around..
GREEN=\x1b\x5b32m
YELLOW=\x1b\x5b33m
DEFCOL=\x1b\x5b39m
BLUE=\x1b\x5b34m

all: print $(OBJECT_FILES)

print: 
	@echo -e "$(GREEN)COMPILING $(YELLOW)package $(PACKAGE) $(DEFCOL)"

%.o: %.cpp %.h
	@echo -e "$(GREEN)COMPILING $(YELLOW)$(PACKAGE)/$*.cpp $(BLUE)"
	@$(CC) $(CPPFLAGS) -c $*.cpp
	@printf "$(DEFCOL)"

clean:
	@echo -e "$(GREEN)CLEANING $(YELLOW)$(PACKAGE) $(DEFCOL)"
	@rm -rf *.o *~ 
