#! /bin/bash

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



cat <<EOF
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
   <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
   <title>COVISE Online Documentation</title>
</head>

<body bgcolor="#FFFFFF" link="#0033cc" vlink="#0033cc">

<table border=0 cellspacing=0 cellpadding=0 width="100%" >

<tr bgcolor="#99CCFF">
<td valign="middle">
EOF

echo '<a href="'${up}'index.html">Overview</a> |'
echo '<a href="'${up}'modules/index.html">All Modules</a> |'
echo '<a href="'${up}'tutorial/index.html">Tutorial</a> |'
echo '<a href="'${up}"usersguide/index.html\">User's Guide</a> |"
echo '<a href="'${up}'programmingguide/index.html">Programming Guide</a>'

cat <<EOF
</td>
</tr>

</table>
EOF

if [ "$prev""$next" != "" ]; then
   echo -n '<table border=0 cellspacing=0 cellpadding=0 width="100%" >'
   echo -n '<tr><td align="left" width="10%">'
   if [ "$prev" != "" ]; then
      echo -n '<a href="'${prev}'">Previous</a>'
   fi
   echo -n '</td><td align="center" width="80%">'
else
   echo -n '<center>'
fi

echo -n '<h3>COVISE Online Documentation</h3>'

if [ "$prev""$next" != "" ]; then
   echo -n '</td><td align="right" width="10%">'
   if [ "$next" != "" ]; then
      echo -n '<a href="'${next}'">Next</a>'
   fi
   echo -n '</td></tr></table>'
else
   echo -n '</center>'
fi
echo '<hr>'

