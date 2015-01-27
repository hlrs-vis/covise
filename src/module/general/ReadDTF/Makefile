##########################################################
## Description: Makefile Template for ReadDTF program     ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Andreas Werner              Date: 10.01.2000 ##
## modified for template use by: Alexander Martinez 	Date: 12.09.2003#
##########################################################

# name of the module executable
APPLICATION = ReadDTF

# name of the module category
CATEGORY = Unsupported

# needed for creating distribution tar-files
PROJECT = General

# put all object files that are needed to build the module here,
# e.g.   $(OBJDIR)/object.o

OBJECTS         = Tools/*.o DTF/*.o DTF_Lib/*.o $(OBJDIR)/DataWrapper.o

# define some colors for funny outputs until covise starts to gag around..
GREEN=\x1b\x5b32m
YELLOW=\x1b\x5b33m
RED=\x1b\x5b31m
BLUE=\x1b\x5b34m
DEFCOL=\x1b\x5b39m

# $(APPLICATION) is also the name of the main source and include file!!!)

##########################################################

# All system-dependent UNIX compiler flags and options are in
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
	@echo -e "$(GREEN)LINKING $(YELLOW)$(PLATFORM)/bin/$(CATEGORY)/$(APPLICATION) $(BLUE)" 
	@$(CC) $(CPPFLAGS) -Wno-deprecated -I./DTF_Lib/cfdrc/include -o $@ \
$(OBJECTS) $(OBJDIR)/$(APPLICATION).o $(COVISE_LIBDIR) \
$(COVISEAPI) $(APPLLIB) $(COVISELIB) -L./DTF_Lib/cfdrc/lib -lDTF -lm
	@echo -e "$(GREEN)LINKING $(YELLOW)$(APPLICATION) $(BLUE)" 
	@$(CC) $(CPPFLAGS) -Wno-deprecated -I./DTF_Lib/cfdrc/include -o $(APPLICATION) \
$(OBJECTS) $(OBJDIR)/$(APPLICATION).o $(COVISE_LIBDIR) \
$(COVISEAPI) $(APPLLIB) $(COVISELIB) -L./DTF_Lib/cfdrc/lib -lDTF -lm
	@printf "$(DEFCOL)"
	@make legend
	
############### general object rule ######################
# can be modified (e.g. for dependencies)
##########################################################

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp $(APPLICATION).h
	@echo -e "$(GREEN)COMPILING $(YELLOW)$(APPLICATION).cpp $(RED)"
	@$(CC) $(CPPFLAGS) -Wno-deprecated -I./DTF_Lib/cfdrc/include -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	@$(POSTCC)
	@printf "$(DEFCOL)"

############### module specific rules ####################
# rules for objects in $(OBJECTS)
##########################################################

$(OBJDIR)/DataWrapper.o: DataWrapper.cpp DataWrapper.h 
	@echo -e "$(GREEN)COMPILING $(YELLOW)DataWrapper.cpp $(RED)" 
	@$(CC) $(CPPFLAGS) -Wno-deprecated -I./DTF_Lib/cfdrc/include -c DataWrapper.cpp -o $(OBJDIR)/DataWrapper.o
	@$(POSTCC)
	@printf "$(DEFCOL)"

docs: *.h *.cpp 
	@echo -e "$(GREEN)GENERATING $(YELLOW)documentation $(DEFCOL)"
	@doxygen Doxyfile > /dev/null
	@cat doxygen_log | less

allclean: 
	@echo -e "$(GREEN)CLEANING UP $(DEFCOL)"
	@make -s clean > /dev/null
	@rm -f *~ doxygen_log
	@echo -e "$(GREEN)CLEANING $(YELLOW)Tools $(DEFCOL)"
	@make -s -C Tools clean > /dev/null
	@echo -e "$(GREEN)CLEANING $(YELLOW)DTF $(DEFCOL)"
	@make -s -C DTF clean > /dev/null
	@echo -e "$(GREEN)CLEANING $(YELLOW)DTF_Lib $(DEFCOL)"
	@make -s -C DTF_Lib clean > /dev/null

Tools/*.o: 
	@make -s -C Tools

DTF/*.o:
	@make -s -C DTF

DTF_Lib/*.o: 
	@make -s -C DTF_Lib

legend: 
	@echo -e ""
	@echo -e "==========================================="
	@echo -e "Description of used colors:"
	@echo -e "==========================================="
	@echo -e "$(GREEN)green  $(DEFCOL) = state (eg. COMPILING)"
	@echo -e "$(YELLOW)yellow $(DEFCOL) = processed file or package"
	@echo -e "$(RED)red    $(DEFCOL) = outputs created by g++ due to COVISE includes.."
	@echo -e "$(BLUE)blue   $(DEFCOL) = normal g++ output"
	@echo -e "==========================================="
	@echo ""
