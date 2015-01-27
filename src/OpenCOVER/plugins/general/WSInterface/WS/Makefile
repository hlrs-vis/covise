all: sampledir

.PHONY : clean

sampledir: wscoverH.h envH.h
	touch sampledir
	mkdir -p sample
	mv *.re[qs].xml sample/

wscoverH.h: wscover.h
	$(GSOAP_BINDIR)/soapcpp2 -S -I$(GSOAP_IMPORTDIR) -ni wscover.h

envH.h: env.h
	$(GSOAP_BINDIR)/soapcpp2 -penv -L env.h


clean:
	rm -fr opencoverH.h opencoverC.cpp opencoverStub.h \
	      opencoverCOVERService.cpp opencoverCOVERService.h \
              env?*.h env*.cpp *.nsmap *.re[qs].xml \
	      COVER.wsdl opencover.xsd \
	      sample sampledir
