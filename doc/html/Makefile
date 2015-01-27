all: subdirs index.html html indices

subdirs:
	cd tutorial && $(MAKE) depend && $(MAKE)
	cd usersguide && $(MAKE) depend && $(MAKE)
	cd programmingguide && $(MAKE) depend && $(MAKE)
	cd cover_inst_config && $(MAKE) depend && $(MAKE)

always_out_of_date:

%.html: %.html.in
	$(COVISEDIR)/doc/scripts/htmlheader.sh 0 > $@
	cat $< >> $@
	$(COVISEDIR)/doc/scripts/htmlfooter.sh 0 >> $@

html:
	echo > $(COVISEDIR)/doc/refguide/modulelist.txt
	cd $(COVISEDIR)/src/module/general && ${COVISEDIR}/doc/scripts/makeallhtml.sh
	cd $(COVISEDIR)/src/module/examples && ${COVISEDIR}/doc/scripts/makeallhtml.sh
	cd $(COVISEDIR)/src/module/koeln && ${COVISEDIR}/doc/scripts/makeallhtml.sh
	cd $(COVISEDIR)/src/module/hlrs && ${COVISEDIR}/doc/scripts/makeallhtml.sh
	cd $(COVISEDIR)/src/module/spp1155 && ${COVISEDIR}/doc/scripts/makeallhtml.sh

indices: index.html always_out_of_date
	cd modules && $(COVISEDIR)/doc/scripts/makeallmod.sh > index.html
	cd modules && dir=`/bin/pwd` && for i in *; do \
	   test -d $$i && echo $$i && cd $$i && $(COVISEDIR)/doc/scripts/makeallmod.sh 1 > index.html; \
	   cd $$dir; \
	done

clean:
	$(RM) index.html
	-cd tutorial && $(MAKE) clean
	-cd usersguide && $(MAKE) clean
	-cd programmingguide && $(MAKE) clean
	-cd cover_inst_config && $(MAKE) clean
	-cd modules && $(RM) index.html
	-cd modules && dir=`/bin/pwd` && for i in *; do \
           test -d $$i && cd $$i && $(RM) *.html; cd $$dir; \
	   test -d $$i && echo $$i && cd $$i && \
              for j in *; do \
                 test -d $$j && echo $$i/$$j && $(RM) -r $$j; cd $$dir/$$i; \
              done; \
	   cd $$dir; \
	done
