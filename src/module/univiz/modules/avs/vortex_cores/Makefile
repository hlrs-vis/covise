#
# Generic Makefile for architecture dependant development
#

AVSMACHINE=linux

.IGNORE:	

all:
	( mkdir -p $(AVSMACHINE) ; \
	rm -f $(AVSMACHINE)/*; \
	cd $(AVSMACHINE) ; \
	ln -s ../* .; \
	rm -f $(AVSMACHINE); \
	gmake -i -f ../Makefile.common )

recompile:
	( cd $(AVSMACHINE) ; rm -f *.o )
	gmake -i all

clean:
	rm -rf linux
	rm ../../../libs/*/*.o
	rm ../../impl/vortex_cores/*.o
