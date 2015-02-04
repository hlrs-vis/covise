#! /bin/bash

DIR=`unset PWD; /bin/pwd`
export NAME=`basename $DIR`
export TARGET=`awk 'BEGIN { FS="= " }; /^[^A-Za-z0-9_]*TARGET[^A-Za-z0-9_]*=/ { print $2 }' < $NAME.pro`
export CATEGORY=`awk 'BEGIN { FS="= " }; /CATEGORY.*=/ { print $2 }' < $NAME.pro`

mkdir -p doc

desc=`bash -c "source $COVISEDIR/scripts/covise-env.sh && $COVISEDIR/$ARCHSUFFIX/bin/$CATEGORY/$TARGET -d" \
        | grep ^Desc: \
        | awk 'BEGIN { FS="\"" }; { printf "%s\n", $2 }' \
        | sed -e 's/_/\\\\_/g'`

echo $desc > doc/shortdesc.tex.in
echo >> doc/shortdesc.tex.in

ALL_ARCHS="win32 gcc3 teck amd64 x64 bishorn heiner gecko skink basilisk waran chuckwalla dapper
hardy heron ia64 sgin32 sgi64 leopard monshuygens lycaeus"
if [ -z "$AVAILABLE_ON" ]; then
   AVAILABLE_ON="$ALL_ARCHS"
fi

if [ ! -z "$NOT_AVAILABLE_ON" ]; then
   for i in $NOT_AVAILABLE_ON; do
        AVAILABLE_ON=`echo $AVAILABLE_ON | sed -e s/$i//`
   done
fi

if [ "$AVAILABLE_ON" = "$ALL_ARCHS" ]; then
   echo "Available on all supported platforms." > doc/available.tex.in
   echo >> doc/available.tex.in
else
   echo -n "Available on " > doc/available.tex.in
   for i in $AVAILABLE_ON; do
       if [ ! -z "$first" ]; then
           echo -n ", " >> doc/available.tex.in
       fi
       case "$i" in
           win32)
                echo -n "x86 Windows 2000/XP (win32)" >> doc/available.tex.in
           ;;
           vista)
                echo -n "x86 Windows XP/Vista (vista)" >> doc/available.tex.in
           ;;
           windows)
                echo -n "Windows" >> doc/available.tex.in
           ;;
           linux)
                echo -n "x86 Red Hat Linux 7.2 (linux)" >> doc/available.tex.in
           ;;
           gcc3)
                echo -n "x86 Red Hat Linux 8.0 (gcc3)" >> doc/available.tex.in
           ;;
           teck)
                echo -n "x86 Fedora Linux Core 3 (teck)" >> doc/available.tex.in
           ;;
           gcc4)
                echo -n "x86 Fedora Linux 3 with GCC 4 (gcc4)" >> doc/available.tex.in
           ;;
           amd64)
                echo -n "x86\\_64 Fedora Linux Core 3 (amd64)" >> doc/available.tex.in
           ;;
           ia64)
                echo -n "ia64 Red Hat Enterprise Linux 3 (ia64)" >> doc/available.tex.in
           ;;
           sgin32)
                echo -n "32 bit IRIX 6.5 (sgin32)" >> doc/available.tex.in
           ;;
           sgi64)
                echo -n "64 bit IRIX 6.5 (sgi64)" >> doc/available.tex.in
           ;;
           macx)
                echo -n "Mac OS X Panther 10.3 (macx)" >> doc/available.tex.in
           ;;
           tiger)
                echo -n "32 bit x86/ppc Mac OS X Tiger 10.4 (tiger)" >> doc/available.tex.in
           ;;
           leopard)
                echo -n "32 bit x86/ppc Mac OS X Leopard 10.5 (leopard)" >> doc/available.tex.in
           ;;
        esac
       first=false
   done
   echo -n "." >> doc/available.tex.in >> doc/available.tex.in
   echo >> doc/available.tex.in >> doc/available.tex.in
fi

echo available: $AVAILABLE_ON

OLDIFS=$IFS
IFS=" "
parameters=`bash -c "source $COVISEDIR/scripts/covise-env.sh && $COVISEDIR/$ARCHSUFFIX/bin/$CATEGORY/$TARGET -d" \
        | sed -e '1,/^Parameters:/d; /^OutPorts:/,$d' \
        | grep -v '__filter.*BrowserFilter' \
        | sed -e 's/_/\\\\_/g' \
        | sed -e 's/%/\\\\%/g' \
        | sed -e 's/|/\\\\newline /g' \
        | awk 'BEGIN { FS="\"" }; { printf "\\\\hline\n\t%s & %s & %s\\\\\\\\ \n", $2, $4, $8 }'`

outports=`bash -c "source $COVISEDIR/scripts/covise-env.sh && $COVISEDIR/$ARCHSUFFIX/bin/$CATEGORY/$TARGET -d" \
        | sed -e '1,/^OutPorts:/d; /^InPorts:/,$d' \
        | sed -e 's/_/\\\\_/g' \
        | sed -e 's/%/\\\\%/g' \
        | sed -e 's/|/\\\\newline /g' \
        | awk 'BEGIN { FS="\"" }; { printf "\\\\hline\n\t%s & %s & %s\\\\\\\\ \n", $2, $4, $6 }'`

inports=`bash -c "source $COVISEDIR/scripts/covise-env.sh && $COVISEDIR/$ARCHSUFFIX/bin/$CATEGORY/$TARGET -d" \
        | sed -e '1,/^InPorts:/d' \
        | sed -e '/^InPorts:/,$d' \
        | sed -e 's/_/\\\\_/g' \
        | sed -e 's/%/\\\\%/g' \
        | sed -e 's/|/\\\\newline /g' \
        | awk 'BEGIN { FS="\"" }; { printf "\\\\hline\n\t%s & %s & %s\\\\\\\\ \n", $2, $4, $6 }'`

tableheader='\begin{longtable}{|p{4cm}|p{4cm}|p{9cm}|} \hline \bf{Name} & \bf{Type} & \bf{Description} \endhead \hline\hline'
tablefooter='\hline \end{longtable}'
tableheaderp='\begin{longtable}{|p{4cm}|p{4cm}|p{9cm}|} \hline \bf{Name} & \bf{Type(s)} & \bf{Description} \endhead \hline\hline'

if [ ! -z "$parameters" ]; then
        echo '\subsubsection{Parameters}' > doc/parameters.tex.in
        echo >> doc/parameters.tex.in
        echo $tableheader >> doc/parameters.tex.in
        echo $parameters >> doc/parameters.tex.in
        echo $tablefooter >> doc/parameters.tex.in
        echo >> doc/parameters.tex.in
else
        rm -f doc/parameters.tex.in
fi

if [ ! -z "$inports" ]; then
        echo '\subsubsection{Input Ports}' > doc/inports.tex.in
        echo >> doc/inports.tex.in
        echo $tableheaderp >> doc/inports.tex.in
        echo $inports >> doc/inports.tex.in
        echo $tablefooter >> doc/inports.tex.in
        echo >> doc/inports.tex.in
else
        rm -f doc/inports.tex.in
fi

if [ ! -z "$outports" ]; then
        echo '\subsubsection{Output Ports}' > doc/outports.tex.in
        echo >> doc/outports.tex.in
        echo $tableheaderp >> doc/outports.tex.in
        echo $outports >> doc/outports.tex.in
        echo $tablefooter >> doc/outports.tex.in
        echo >> doc/outports.tex.in
else
        rm -f doc/outports.tex.in
fi
