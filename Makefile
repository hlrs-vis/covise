all: covise cover-dep addons-dep python-dep

always_out_of_date:

verbose: always_out_of_date
	cd src && $(MAKE) verbose
	cd src/OpenCOVER && $(MAKE) verbose
	cd Python && $(MAKE)

bin: covise cover-dep addons-dep

python-dep: covise cover-dep addons-dep
	$(MAKE) python

python:
	cd Python && $(MAKE)

covise: always_out_of_date
	cd src && $(MAKE)

cover-dep: covise
	$(MAKE) cover

cover:
	cd src/OpenCOVER && $(MAKE)

addons-dep: covise cover-dep
	$(MAKE) addons

addons:
	@for dir in $${COVISE_ADDONS}; do \
		export BUILDDIR=$${COVISEDIR}/$${ARCHSUFFIX}/build.addon-$$(basename $${dir}); \
		mkdir -p $${BUILDDIR} && \
		cd $${BUILDDIR} && \
		cmake $${dir} && \
		make; \
	done

doc:	always_out_of_date
	cd doc && $(MAKE) html
	cd doc && $(MAKE) pdf
	cd doc && $(MAKE) doxygen

docclean:
	cd doc && $(MAKE) clean

package: dist
archive: dist

dist:   always_out_of_date
	cd archive && $(MAKE)

archdist:	always_out_of_date
	cd archive && $(MAKE) arch

shareddist:	always_out_of_date
	cd archive && $(MAKE) shared

install: always_out_of_date
	$(MAKE) -f src/Makefile.default install
	cd src/OpenCOVER && $(MAKE) install
	
clean:
	cd src && $(MAKE) clean
	cd Python && $(MAKE) clean
	test -n "$${ARCHSUFFIX}" && $(RM) -rf "$${ARCHSUFFIX}/lib" "$${ARCHSUFFIX}/bin" || exit 0

makefiles depend:
	cd src && $(MAKE) makefiles

indent:
	test -n "$${COVISEDIR}" && cd src/kernel/api && find . \( -name "*.h" -o -name "*.c" -o -name "*.cpp" \) -print -exec ${COVISEDIR}/script/indent.sh \{\} \;

help:
	@echo 'available make targets:'
	@echo '  makefiles    recreate all Makefile.'$${ARCHSUFFIX}
	@echo '  bin          compile everything'
	@echo '  clean        remove everything built by "bin"'
	@echo '  doc          generate pdf & html documentation'
	@echo '  docclean     remove everything built by "doc"'
	@echo '  dist         archive SHARED and '$${ARCHSUFFIX}' parts'
	@echo '  shareddist   archive SHARED parts'
	@echo '  archdist     archive '$${ARCHSUFFIX}' parts'
	@echo '  indent       reindent source files'
