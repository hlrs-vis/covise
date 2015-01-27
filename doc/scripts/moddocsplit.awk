#! /usr/bin/awk -f

BEGIN { sec=0; }

sec>0  && /^ *\\subsubsection{.*}/ { sec=5; print "" > "aftertable.tex.in" }

sec==1                    { print >> "beforetable.tex.in" }
sec==0 && /^ *\\label{.*} *$/ { sec=1; print "" > "beforetable.tex.in" }

sec>0  && /^ *\\subsubsection{Parameters}/ { sec=2; print "" > "parameters.tex.in" }
sec==2                                   { print >> "parameters.tex.in" }

sec>0  && /^ *\\subsubsection{Input Ports}/ { sec=3; print "" > "inports.tex.in" }
sec==3                                   { print >> "inports.tex.in" }

sec>0  && /^ *\\subsubsection{Output Ports}/ { sec=4; print "" > "outports.tex.in" }
sec==4                                   { print >> "outports.tex.in" }

sec==5                                   { print >> "aftertable.tex.in" }
