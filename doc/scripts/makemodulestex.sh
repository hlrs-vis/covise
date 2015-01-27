#! /bin/bash

cat modules.tex.header

sect=""
for i in `cat modulelist.txt | sort -u`; do
        nsect=`echo "$i" | cut -f1 -d:`
	if [ "$nsect" != "" ]; then
        	if [ "$sect" != "$nsect" ]; then
                	if [ "$sect" != "" ]; then
                        	echo '\newpage'
                	fi
                	sect="$nsect"
                	echo '\section{' `echo "$sect"|sed -e 's/_/\\\\_/g'` '}'
                	echo '\label{' "$sect" '}'
        	fi
        	tex=`echo "$i" | cut -f2 -d: | sed -e 's/ //g'`
        	echo '\graphicspath{{'`echo "$tex" | sed -e 's,/[^/]*$,,'`/'}}'
        	echo '\input{'"$tex"'}'
        	echo '\clearpage'
	fi
done
