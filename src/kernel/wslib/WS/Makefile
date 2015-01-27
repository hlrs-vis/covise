all: sampledir

.PHONY : clean

sampledir: coviseH.h envH.h 
	touch sampledir
	mkdir -p sample
	mv *.re[qs].xml sample/

coviseH.h: wscovise.h wscoviseevents.h
	$(GSOAP_BINDIR)/soapcpp2 -S -I$(GSOAP_IMPORTDIR) -ni -t wscovise.h
	sed -i -e 's/struct soap \*\/\*soap\*\//struct soap \* soap/g' coviseC.cpp
	rm -f $(COVISEDIR)/$(ARCHSUFFIX)/build/sys/wsinterface/qrc_wsdl.cpp
	rm -f $(COVISEDIR)/$(ARCHSUFFIX)/build/sys/wsinterface/qrc_wsdl.o

envH.h: env.h
	$(GSOAP_BINDIR)/soapcpp2 -penv -L env.h
	sed -i -e 's/struct soap \*\/\*soap\*\//struct soap \* soap/g' envC.cpp

clean:
	rm -fr \
	      coviseH.h coviseC.cpp coviseStub.h \
	      coviseeventsH.h coviseeventsC.cpp coviseeventsStub.h \
	      coviseCOVISEService.cpp coviseCOVISEService.h \
	      coviseCoviseEventsService.cpp coviseCoviseEventsService.h \
	      coviseeventsCoviseEventsProxy.cpp coviseeventsCoviseEventsProxy.h \
	      env?*.h env*.cpp *.nsmap *.re[qs].xml \
	      COVISE.wsdl covise.xsd \
	      CoviseEvents.wsdl coviseevents.xsd \
	      sample sampledir
