all: flowvis_modules_cgl.aux flowvis_modules_cgl.bbl flowvis_modules_cgl.tex intro.tex vortex_cores.tex
#	pdflatex flowvis_modules_cgl.tex
	latex flowvis_modules_cgl.tex
	dvipdfm flowvis_modules_cgl

flowvis_modules_cgl.aux: flowvis_modules_cgl.tex
#	pdflatex flowvis_modules_cgl.tex
	latex flowvis_modules_cgl.tex

flowvis_modules_cgl.bbl: flowvis_modules_cgl.bib flowvis_modules_cgl.tex
#	rm flowvis_modules_cgl.bbl
	bibtex flowvis_modules_cgl

clean:
#	rm *.aux *.pdf *.log *.toc *.out *.bbl *.blg *.brf *.dvi
	rm *.aux *.pdf *.log *.toc *.bbl *.blg *.brf *.dvi
	rm -rf flowvis_modules_cgl/

html:
	make all
	make all
#	latex2html -toc_depth 1 flowvis_modules_cgl.tex
#	latex2html -split 3 flowvis_modules_cgl.tex
#	latex2html -split 3 -no_math -html_version 3.2,math flowvis_modules_cgl.tex
#	latex2html -toc_depth 2 -contents_in_navigation -split 3 -local_icons -no_math -html_version 3.2,math flowvis_modules_cgl.tex
#	latex2html -toc_depth 2 -split 3 -contents_in_navigation -noinfo -local_icons -no_math -html_version 3.2,math flowvis_modules_cgl.tex
	latex2html -toc_depth 3 -split 4 -contents_in_navigation -noinfo -local_icons -no_math -html_version 3.2,math flowvis_modules_cgl.tex
