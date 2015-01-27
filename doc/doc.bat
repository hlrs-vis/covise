rem
rem Windows batch file "doc.bat"
rem creates the covise pdf's
rem
rem
echo off
if not exist pdf mkdir pdf
echo Generating Refguide...
cd refguide
latex refguide.tex
latex refguide.tex
dvips refguide.dvi
ps2pdf refguide.ps
copy refguide.pdf ..\pdf
cd ..
echo ...finished.
echo Generating Tutorial...
cd Tutorial
latex Tutorial.tex
latex Tutorial.tex
dvips Tutorial.dvi
ps2pdf Tutorial.ps
copy Tutorial.pdf ..\pdf
cd ..
echo ...finished.
echo Generating Programming Guide...
cd programmingguide
latex programmingguide.tex
latex programmingguide.tex
dvips programmingguide.dvi
ps2pdf programmingguide.ps
copy programmingguide.pdf ..\pdf
cd ..
echo ...finished.
echo Generating Cover Inst Config...
cd cover_inst_config
latex cover_inst_config.tex
latex cover_inst_config.tex
dvips cover_inst_config.dvi
ps2pdf cover_inst_config.ps
copy cover_inst_config.pdf ..\pdf
cd ..
echo ...finished.
echo ...script finished.
echo on




