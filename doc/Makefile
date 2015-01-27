
DOCU     = html pdf doxygen tips


all: $(DOCU) 

always_out_of_date:

pdf/%.pdf: % always_out_of_date
	cd $< && $(MAKE) all

%html: % always_out_of_date
	cd $< && $(MAKE) $@

pdf:	pdf/programmingguide.pdf \
      pdf/cover_inst_config.pdf \
	   pdf/tutorial.pdf \
	   pdf/usersguide.pdf \
	   pdf/refguide.pdf 

html:   always_out_of_date
	cd html && $(MAKE)

doxygen:   always_out_of_date
	cd doxygen && $(MAKE)

clean:
	$(RM) index.html
	cd html && $(MAKE) clean

tips:
	$(MAKE) -C tooltips
