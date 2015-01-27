#! /bin/bash

level="$1"

if [ -z "$level" ]; then
   level=0
fi

${COVISEDIR}/doc/scripts/htmlheader.sh $(( $level+1 ))

if [ -f index.html.in ]; then
    cat index.html.in
fi

nmod=`find . -name '*.html' -print | wc | awk '{ print $1 }'`
ncol=4

echo '<table border="0" cellspacing="0" cellpadding="0" width="100%">'

keyind=$(( 3-$level ))

list=`find . -name '*.html' -print \
     | grep -v '/index.html$' \
     | cut -d / -f -${keyind} \
     | sort -f -t / -k $keyind -u`

nmod=`echo $list | wc | awk '{print $2}'`

lines=$(( ($nmod+$ncol-1)/$ncol ))

max=$(( $lines*$ncol ))

count=0

prev=""
for i in $list; do
        module[$count]=$i

        letter[$count]=`echo $i|cut -d/ -f$keyind | cut -b1 | tr a-z A-Z`
        if [ "$prev" = ${letter[$count]} ]; then
             letter[$count]=""
        else
             prev=${letter[$count]}
             letter[$count]='<b>'$prev'</b>'
        fi

        count=$(( $count+1 ))
done

for((count=0; $count < $max; count++)); do
        if [ "$(( $count%$ncol ))" = "0" ]; then
                echo '<tr bgcolor="#ffffcc">'
        fi
        ind=$(( $count%$ncol*$lines + $count/$ncol ))
        echo -n '<td align="right">'${letter[$ind]}'</td>'
        if [ "$level" = "0" ]; then
           echo ${module[$ind]} \
           | awk 'BEGIN {FS="/"}; {printf "<td><a href=\"%s/%s/%s.html\">%s</a></td>", $2, $3, $3, $3}'
        elif [ "$level" = "1" ]; then
           echo ${module[$ind]} \
           | awk 'BEGIN {FS="/"}; {printf "<td><a href=\"%s/%s.html\">%s</a></td>", $2, $2, $2}'
        fi
        if [ "$(( $count%$ncol ))" = "$(( $ncol-1 ))" ]; then
                echo '</tr>'
        fi
done

echo '</table>'

${COVISEDIR}/doc/scripts/htmlfooter.sh $(( level+1 ))

