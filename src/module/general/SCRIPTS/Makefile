APPLICATION = fluent2coviseASC
PROJECT	    = General

$(COVISEDIR)/bin/$(APPLICATION) : $(APPLICATION)
	-/bin/rm -f $@
	cp $(APPLICATION) $@
	chmod u+w $@
	echo \# `$$COVISEDIR/src/make.ident | cut -f2 -d\"` >> $@
	chmod 555 $@

always_out_of_date:

tar_bin_modules: /view/main_latest/vobs/covise/DIST/DIST.$(ARCHSUFFIX) always_out_of_date
	@/view/main_latest/vobs/covise/mkspecs/makearchive.sh /view/main_latest/vobs/covise/DIST/DIST.$(ARCHSUFFIX)/co$(ARCHSUFFIX).modules.General.tar $(COVISEDIR)/bin/$(APPLICATION)
