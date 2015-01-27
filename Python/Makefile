##########################################################################################
#
# this makefile copies the file scriptInterface.py to the adequate place
#
# (C) 2002 VirCinity GmbH, Stuttgart   info@vircinity.com
#
# 31.10.2002 [RM]
#
##########################################################################################


SCRIPTINTERFACE=$(COVISEDIR)/$(ARCHSUFFIX)/bin/scriptInterface
CO_PYWRAPPER=coPyModules.py
ifdef PYTHON_HOME
PYTHON=$(PYTHON_HOME)/bin/python3
else
PYTHON=python3
endif


all:
	$(MAKE) clean
	$(MAKE) $(SCRIPTINTERFACE)
	$(MAKE) $(CO_PYWRAPPER)
	$(MAKE) vr-prepare
	$(MAKE) pyc

vr-prepare:
	PYTHON=$(PYTHON) ./makeVrPrepare.sh

pyc:
	@echo "Creating pyc files"
	$(PYTHON) -m compileall .


$(SCRIPTINTERFACE) : scriptInterface 
	mkdir -p $(COVISEDIR)/$(ARCHSUFFIX)/bin
	cp ./scriptInterface $(SCRIPTINTERFACE)
	chmod ug+x  $(SCRIPTINTERFACE)

$(CO_PYWRAPPER) :
	PYTHON=$(PYTHON) ./makeBasiModCode.sh > $(CO_PYWRAPPER) || $(RM) $(CO_PYWRAPPER)


clean:
	if [ -e $(SCRIPTINTERFACE) ]; then rm -f $(SCRIPTINTERFACE); fi;
	if [ -e ./$(CO_PYWRAPPER) ]; then rm -f ./$(CO_PYWRAPPER); fi
	if [ -e ./core ]; then rm -f ./core; fi	
	$(RM) ./covise_module_stub_gen_report*
	PYTHONPATH=$(COVISEDIR)/$(ARCHSUFFIX)/lib $(MAKE) -C bin/vr-prepare clean
