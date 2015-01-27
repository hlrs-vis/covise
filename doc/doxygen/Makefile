all: doxydoc

always_out_of_date:

%.html: %.html.in
	$(COVISEDIR)/../common/mkspecs/htmlheader.sh 0 > $@
	cat $< >> $@
	$(COVISEDIR)/../common/mkspecs/htmlfooter.sh 0 >> $@

doxydoc:
	cd $(COVISEDIR)/src/kernel && doxygen
	cd $(COVISEDIR)/src/OpenCOVER && doxygen
	cd $(COVISEDIR)/src/OpenCOVER/plugins/general/Vrml97 && doxygen
	cd $(COVISEDIR)/src/sys/mapeditor && doxygen

clean:
	$(RM) */*
