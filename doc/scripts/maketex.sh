#! /bin/bash

if [ "$1" = "RENDER" ]; then
   shift
fi

#out:
#default
#dep Data

#in:
#req
#opt

RELCOVDIR=`unset PWD; /bin/pwd | sed -e "s,^$COVISEDIR,,"`
case "$RELCOVDIR" in
   /*)
   ;;
   *)
   RELCOVDIR=/"$RELCOVDIR"
   ;;
esac
RELCOVDIR=`echo $RELCOVDIR | sed -e 's,/[^/][^/]*,/..,g' | sed -e s,^/,,`

echo '\begin{htmlonly}'
echo '\input{'$RELCOVDIR/doc/htmlinc'}'
echo '\end{htmlonly}'
echo '\startdocument'
echo '\tableofchildlinks'

echo "\\subsection{`echo $NAME|sed -e 's/_/\\\\_/g'`}"
echo "\\label{$NAME}"

#desc=`tcsh -c "source $COVISEDIR/.env && $COVISEDIR/$ARCHSUFFIX/bin/$CATEGORY/$NAME -d" \ | grep ^Desc: \ | awk 'BEGIN { FS="\"" }; { printf "%s\n", $2 }'`

if [ -f beforetable.tex.in ]; then
    cat beforetable.tex.in
else
    if [ -f shortdesc.tex.in ]; then
        cat shortdesc.tex.in
    else
        echo $desc
    fi

    if [ -f available.tex.in ]; then
        cat available.tex.in
    fi
fi

#parameters=`tcsh -c "source $COVISEDIR/.env && $COVISEDIR/$ARCHSUFFIX/bin/$CATEGORY/$NAME -d" \ | sed -e '0,/^Parameters:/d; /^OutPorts:/,$d' \ | sed -e 's/_/\\\\_/g' \ | sed -e 's/|/ /g' \ | awk 'BEGIN { FS="\"" }; { printf "\\\\hline\n\t%s & %s & %s\\\\\\\\ \n", $2, $4, $8 }'`

#outports=`tcsh -c "source $COVISEDIR/.env && $COVISEDIR/$ARCHSUFFIX/bin/$CATEGORY/$NAME -d" \ | sed -e '0,/^OutPorts:/d; /^InPorts:/,$d' \ | sed -e 's/_/\\\\_/g' \ | sed -e 's/|/ /g' \ | awk 'BEGIN { FS="\"" }; { printf "\\\\hline\n\t%s & %s & %s\\\\\\\\ \n", $2, $4, $6 }'`

#inports=`tcsh -c "source $COVISEDIR/.env && $COVISEDIR/$ARCHSUFFIX/bin/$CATEGORY/$NAME -d" \ | sed -e '0,/^InPorts:/d; /^InPorts:/,$d' \ | sed -e 's/_/\\\\_/g' \ | sed -e 's/|/ /g' \ | awk 'BEGIN { FS="\"" }; { printf "\\\\hline\n\t%s & %s & %s\\\\\\\\ \n", $2, $4, $6 }'`

tableheader='\begin{longtable}{|p{2.5cm}|p{4cm}|p{7cm}|} \hline \bf{Name} & \bf{Type} & \bf{Description} \endhead \hline\hline'
tablefooter='\hline \end{longtable}'
tableheaderp='\begin{longtable}{|p{2.5cm}|p{4cm}|p{7cm}|} \hline \bf{Name} & \bf{Type(s)} & \bf{Description} \endhead \hline\hline'

if [ -f parameters.tex.in ]; then
    cat parameters.tex.in
fi

if [ -f inports.tex.in ]; then
    cat inports.tex.in
fi

if [ -f outports.tex.in ]; then
    cat outports.tex.in
fi

if [ -f aftertable.tex.in ]; then
    cat aftertable.tex.in
fi
