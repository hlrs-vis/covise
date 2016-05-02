BASEDIR=${COVISEDIR}
INSTSCRIPT=covinst.sh
DMG=$(DIST)/covise-${ARCHSUFFIX}.dmg

FILESERVER=fs.hlrs.de
FILESERVERDIR=/var/www/projects/covise/support/download
RAIDSERVER=visent.hlrs.de
RAIDDIR=/mnt/raid/var/www/support/download

AUTOGENH= ${COVISEDIR}/${ARCHSUFFIX}/build.covise/src/kernel/util/covise_version_autogen.h
MAJOR=`grep "COVISE_VERSION_MAJOR" <  ${AUTOGENH} |  cut -d ' ' -f 3`
MINOR=`grep "COVISE_VERSION_MINOR" <  ${AUTOGENH} |  cut -d ' ' -f 3`
PATCH=`grep "COVISE_VERSION_PATCH" <  ${AUTOGENH} |  cut -d ' ' -f 3`
HASH=`grep "COVISE_VERSION_HASH" <  ${AUTOGENH} |  cut -d '"' -f 2`
DATE=`grep "COVISE_VERSION_DATE" <  ${AUTOGENH} |  cut -d '"' -f 2`
#VERSION=${MAJOR}.${MINOR}.${PATCH}`grep "COVISE_VERSION_REVISION" <  ${AUTOGENH} |  cut -d '"' -f 2`
VERSION="${DATE}"
BUILDNUMBER=`date +%y%m%d%H%M`

DIST=$(BASEDIR)/DIST
ARCHDIST=$(DIST)/DIST.${ARCHSUFFIX}
SHAREDIST=$(DIST)/SHARED

ifeq ($(shell uname), Darwin)
all: dmg
else
all: arch shared
endif
arch: bindist platform extlibdist instscript
shared: instscript sharedist docdist datadist tarshared

always_out_of_date:


instscript: $(DIST)/$(INSTSCRIPT)

$(DIST)/$(INSTSCRIPT): $(BASEDIR)/archive/$(INISTSCRIPT) $(DIST) always_out_of_date
	$(RM) $(DIST)/$(INSTSCRIPT); \
	sed -e "s/^ *COVISE_VERSION=.*/ COVISE_VERSION=\"$(MAJOR).$(MINOR)\"/" \
           $(BASEDIR)/archive/$(INSTSCRIPT) > $(DIST)/$(INSTSCRIPT) && chmod 755 $(DIST)/$(INSTSCRIPT)


platform: $(ARCHDIST)/Platform

$(ARCHDIST)/Platform: $(ARCHDIST) always_out_of_date
	$(BASEDIR)/archive/generate-platform.sh



sharedist: $(SHAREDIST)
	cd $(BASEDIR) && $(BASEDIR)/archive/archiveexpand.sh $(SHAREDIST)/coshared.tar $(BASEDIR)/archive/add-sharedist-files.txt

bindist: $(ARCHDIST) instscript platform
	@if [ "`uname`" = "Darwin" ]; then \
		echo "Retrieving non-standard system libraries..."; \
		$(BASEDIR)/archive/systemlibs-mac.sh $(BASEDIR)/${ARCHSUFFIX}/bin/* $(BASEDIR)/${ARCHSUFFIX}/bin/*/* $(BASEDIR)/${ARCHSUFFIX}/lib/OpenCOVER/plugins/* $(BASEDIR)/${ARCHSUFFIX}/bin/*.app/Contents/MacOS/*; \
	fi
	@echo "Retrieving list of installed packages:"
	@if which yum > /dev/null; then yum list installed > ../$(ARCHSUFFIX)/rpm-package-list.txt; fi
	@echo "Retrieving list of installed packages done."
	@echo "Collecting binary files:"
	@cd .. && find $(ARCHSUFFIX)/bin $(ARCHSUFFIX)/lib $(ARCHSUFFIX)/system-lib $(ARCHSUFFIX)/CoviseOptions.cmake $(ARCHSUFFIX)/rpm-package-list.txt -print0 | tar --null --no-recursion -czvf $(ARCHDIST)/co$(ARCHSUFFIX).bin.tar.gz --files-from -
	@echo "Collecting binary files done."

raidarch:
	@echo Covise version is: $(VERSION)
	@ssh $(RAIDSERVER) mkdir -p $(RAIDDIR)/snapshot-$(VERSION)
	@echo creating $(ARCHSUFFIX).tar ...
	@tar -C .. -cvf - DIST/DIST.$(ARCHSUFFIX) | ssh $(RAIDSERVER) "cat - > $(RAIDDIR)/snapshot-$(VERSION)/$(ARCHSUFFIX).tar"

raidshared:
	@echo Covise version is: $(VERSION)
	@ssh $(RAIDSERVER) mkdir -p $(RAIDDIR)/snapshot-$(VERSION)
	@echo creating shared.tar ...
	@tar -C .. -cvf - DIST/SHARED DIST/${INSTSCRIPT} | ssh $(RAIDSERVER) "cat - > $(RAIDDIR)/snapshot-$(VERSION)/shared.tar"

raiddmg: $(DMG)
	@echo Covise version is: $(VERSION)
	@ssh $(RAIDSERVER) mkdir -p $(RAIDDIR)/snapshot-$(VERSION)
	@scp $(DMG) $(RAIDSERVER):$(RAIDDIR)/snapshot-$(VERSION)/

raidindex:
	@ssh $(RAIDSERVER) mkdir -p $(RAIDDIR)/snapshot-$(VERSION)
	@scp ../DIST/${INSTSCRIPT} $(RAIDSERVER):$(RAIDDIR)/snapshot-$(VERSION)
	@echo creating index for $(RAIDDIR)/snapshot-$(VERSION)
	@ssh $(RAIDSERVER) "cd $(RAIDDIR)/snapshot-$(VERSION); ../genindex.pl > index.html; cd $(RAIDDIR); ./genindex-global.pl > index.html"

raidsync:
	-ssh $(RAIDSERVER) "rsync -avz $(RAIDDIR)/ $(FILESERVER):$(FILESERVERDIR) 2> /dev/null"
	@echo done sync

ifeq ($(shell uname), Darwin)
www: raiddmg raidindex raidsync
else
www: raidarch raidshared raidindex raidsync
endif

srcdist:
	cd ../src && $(MAKE) srcdist


datadist: exampledist tutorialdist


docdist: docpdfdist dochtmldist docdoxygendist


doc%dist: % $(SHAREDIST)
	@export DOCDIST=true; cd $(BASEDIR) && $(BASEDIR)/archive/archiveexpand.sh $(SHAREDIST)/codoc.$<.tar $(BASEDIR)/archive/add-docdist$<-files.txt

pdf html doxygen: always_out_of_date


exampledist: $(SHAREDIST)
	cd $(BASEDIR) && $(BASEDIR)/archive/archiveexpand.sh $(SHAREDIST)/coexample.maps_data.tar $(BASEDIR)/archive/add-exampledist-files.txt

tutorialdist: $(SHAREDIST)
	cd $(BASEDIR) && $(BASEDIR)/archive/archiveexpand.sh $(SHAREDIST)/cotutorial.maps_data.tar $(BASEDIR)/archive/add-tutorialdist-files.txt


extlibdist: $(ARCHDIST)
	@printf "Archiving 3rd party libraries:"
	@cd ${EXTERNLIBS}/../.. && \
	   for i in `cat $(BASEDIR)/archive/extlibs-${ARCHSUFFIX:%opt=%}-files.txt` ALL; do \
	      if [ -d "extern_libs/${ARCHSUFFIX}" ]; then \
	         EXTLIBARCH="${ARCHSUFFIX}"; \
              else \
	         EXTLIBARCH="${ARCHSUFFIX:%opt=%}"; \
              fi; \
	      if [ -d "extern_libs/$${EXTLIBARCH}/$$i" ]; then \
	      	 echo "extern_libs/$${EXTLIBARCH}/$$i"/* >> $(BASEDIR)/extlibfilelist.$$$$.txt; \
	       else \
	         echo "extern_libs/$${EXTLIBARCH}/$$i" >> $(BASEDIR)/extlibfilelist.$$$$.txt; \
	       fi; \
           done && \
		if [ -f $(BASEDIR)/extlibfilelist.$$$$.txt ]; then \
		    tar -cf - `cat $(BASEDIR)/extlibfilelist.$$$$.txt` \
               | gzip -c > $(ARCHDIST)/co${ARCHSUFFIX}.extern_libs.tar.gz; \
			 printf " "`ls -sk1 "$(ARCHDIST)/co${ARCHSUFFIX}.extern_libs.tar.gz"| awk '{ print $$1 }'`K 1>&2; \
          $(RM) $(BASEDIR)/extlibfilelist.$$$$.txt; \
	   fi
	@echo .

$(ARCHDIST): $(DIST)
	mkdir -p $@

$(SHAREDIST): $(DIST)
	mkdir -p $@

$(DIST):
	mkdir -p $@

%.tar: %.tar.filelist-base %.tar.filelist-extlib
	@empty=true; \
	for f in $^; do \
	   case $$f in \
	      *.filelist-base) cd $(BASEDIR) ;; \
	      *.filelist-extlib) cd $(EXTERNLIBS)/../.. ;; \
	   esac; \
	   num=`wc -w < "$$f"`; \
	   if [ "$$num" != "0" ]; then \
	      if [ "$$empty" = "true" ]; then \
	         printf "Archiving `basename $@`: $$num" 1>&2 ; \
	      else \
	         printf "+$$num" 1>&2 ; \
	      fi ; \
	      empty=false; \
	      if [ -f "$@" ]; then \
	         tar -r -T "$$f" -f "$@"; \
	      else \
	         tar -c -T "$$f" -f "$@"; \
	      fi; \
	      $(RM) $$f; \
	   fi; \
	done ; \
	if [ "$$empty" != "true" ]; then \
	   printf " files, " 1>&2 ; \
	fi; \
	$(RM) $^


%.tar.gz: %.tar
	@printf "compressing: `ls -sk1 $<| awk '{ print $$1 }'`K ->" 1>&2
	@gzip --fast -f $<
	@echo " "`ls -sk1 "$<.gz"| awk '{ print $$1 }'`K 1>&2

tarshared: sharedist
	@for f in $(SHAREDIST)/*.tar.filelist*; do \
	   case $$f in \
	      *NoArchive*) continue ;; \
	   esac ;\
	   archive=`echo $$f | sed -e s/tar.filelist-.*$$/tar/` ;\
	   $(MAKE) $$archive.gz > /dev/null ;\
	done

dmg: arch shared
	$(RM) $(DMG)
	size=`du -csk $(DIST)/${INSTSCRIPT} $(DIST)/SHARED/ $(ARCHDIST)/ | tail -n 1 | cut -f1`; size=$$(($$size + 250000)); hdiutil create -volname COVISE -size $${size}k -fs HFS+ $(DIST)/covise-${ARCHSUFFIX}.dmg
	hdiutil detach $(DIST)/dmg-${ARCHSUFFIX} || true
	mountpoint=`cd $(DIST)/dmg-${ARCHSUFFIX}; /bin/pwd -P`; hdiutil detach $$mountpoint || true
	mkdir -p $(DIST)/dmg-${ARCHSUFFIX}
	mountpoint=`cd $(DIST)/dmg-${ARCHSUFFIX}; /bin/pwd -P`; hdiutil attach -mountpoint $$mountpoint $(DMG)
	mkdir -p $(DIST)/dmg-${ARCHSUFFIX}/SHARED $(DIST)/dmg-${ARCHSUFFIX}/DIST.${ARCHSUFFIX}
	cp $(ARCHDIST)/Platform $(DIST)/dmg-${ARCHSUFFIX}/DIST.${ARCHSUFFIX}/
	cp $(ARCHDIST)/*.tar.gz $(DIST)/dmg-${ARCHSUFFIX}/DIST.${ARCHSUFFIX}/
	cp $(DIST)/SHARED/*.tar.gz $(DIST)/dmg-${ARCHSUFFIX}/SHARED/
	cp $(DIST)/${INSTSCRIPT} $(DIST)/dmg-${ARCHSUFFIX}/${INSTSCRIPT}
	chmod a+rx $(DIST)/dmg-${ARCHSUFFIX}/${INSTSCRIPT}
	mountpoint=`cd $(DIST)/dmg-${ARCHSUFFIX}; /bin/pwd -P`; hdiutil detach $$mountpoint
	hdiutil checksum -type MD5 $(DMG)
	rmdir $(DIST)/dmg-${ARCHSUFFIX}
