#! /bin/bash

YEAR=`date +%Y`

while [ ! -z "$1" ]; do
   if [ "$1" = "-n" ]; then
      shift
      next="$1"
      shift
   elif [ "$1" = "-p" ]; then
      shift
      prev="$1"
      shift
   else
      level="$1"
      shift
   fi
done

if [ -z "$level" ]; then
        level=2
fi

case $level in
        0) up="" ;;
        1) up="../" ;;
        2) up="../../" ;;
        3) up="../../../" ;;
        4) up="../../../../" ;;
        5) up="../../../../.." ;;
esac

if [ "$prev""$next" != "" ]; then
   echo -n '<table border=0 cellspacing=0 cellpadding=0 width="100%" >'
   echo -n '<tr><td align="left" width="10%">'
   if [ "$prev" != "" ]; then
      echo -n '<a href="'${prev}'">Previous</a>'
   fi
   echo -n '</td><td align="center" width="80%">'

   echo -n '</td><td align="right" width="10%">'
   if [ "$next" != "" ]; then
      echo -n '<a href="'${next}'">Next</a>'
   fi
   echo -n '</td></tr></table>'
fi

cat <<EOF
<hr>
<center><table border=0 cellspacing=0 width="100%" >
<tr>
<td>Authors: Martin Aum&uuml;ller, Ruth Lang, Daniela Rainer, J&uuml;rgen Schulze-D&ouml;bold, Andreas Werner, Peter Wolf, Uwe W&ouml;ssner
</tr>
</table></center>

<center><table border=0 cellspacing=0 width="100%" >
<tr>
<td>Copyright &copy; 1993-$YEAR <a href="http://www.hlrs.de">HLRS</a>, 2004-2014 <a href="http://vis.uni-koeln.de">RRZK</a>, 2005-2014 <a href="http://visenso.com">Visenso</a></td>

<td align=right>
<div align=right>COVISE Version ${COVISE_VERSION}</div>
</td>
</tr>

</table></center>

</body>
</html>
EOF
